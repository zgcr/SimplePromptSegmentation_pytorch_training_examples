from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'UniversalSegmentationJointAccelerateLoss',
]


class Mask2FormerHungarianAccelerateMatcher(nn.Module):

    def __init__(self,
                 mask_cost=1.0,
                 dice_cost=1.0,
                 class_cost=1.0,
                 max_prompt_num=1,
                 slot_constrained=False):
        super(Mask2FormerHungarianAccelerateMatcher, self).__init__()
        self.mask_cost = mask_cost
        self.dice_cost = dice_cost
        self.class_cost = class_cost
        self.max_prompt_num = max_prompt_num
        self.slot_constrained = slot_constrained

        self.sigmoid_ce_loss = nn.BCEWithLogitsLoss(reduction="none")

    @torch.no_grad()
    def forward(self, mask_preds, class_preds, mask_gts, class_gts):
        # num_classes has background class
        # mask_preds:[batch_size, query_nums, height, width]
        # class_preds:[batch_size, query_nums, num_classes]
        # mask_gts[0]:[mask_nums, height, width]
        # class_gts[0]:[mask_nums]

        batch_size, query_nums, H, W = mask_preds.shape
        pixels = H * W
        max_n = max(
            1,
            min(
                max(
                    int(per_image_mask_gts.shape[0])
                    for per_image_mask_gts in mask_gts), query_nums))

        # [B, query_nums, H*W]
        pred_flat = mask_preds.flatten(2)

        target_padded = torch.zeros(batch_size,
                                    max_n,
                                    pixels,
                                    device=pred_flat.device,
                                    dtype=pred_flat.dtype)
        class_targets_padded = torch.zeros(batch_size,
                                           max_n,
                                           dtype=torch.long,
                                           device=pred_flat.device)

        n_masks_per_image = []
        for i in range(batch_size):
            n_total = mask_gts[i].shape[0]
            n = min(n_total, max_n)
            n_masks_per_image.append(n)
            target_padded[i, :n] = mask_gts[i][:n].to(pred_flat).flatten(1)
            class_targets_padded[i, :n] = class_gts[i][:n]

        # batch pairwise CE cost
        ce_pos = F.softplus(-pred_flat) / pixels
        ce_neg = F.softplus(pred_flat) / pixels
        target_padded_T = target_padded.transpose(1, 2)
        mask_cost = torch.bmm(ce_pos, target_padded_T) + \
                    torch.bmm(ce_neg, (1 - target_padded).transpose(1, 2))

        # batch pairwise Dice cost
        pred_sigmoid = torch.sigmoid(pred_flat)
        numerator = 2 * torch.bmm(pred_sigmoid, target_padded_T)
        denominator = pred_sigmoid.sum(dim=-1, keepdim=True) + \
                    target_padded.sum(dim=-1).unsqueeze(1)
        dice_cost = 1 - (numerator + 1) / (denominator + 1)

        # batch class cost
        pred_probs = class_preds.softmax(dim=-1)
        class_cost = -torch.gather(
            pred_probs, 2,
            class_targets_padded.unsqueeze(1).expand(-1, query_nums, -1))

        total_cost = self.mask_cost * mask_cost + \
                    self.dice_cost * dice_cost + \
                    self.class_cost * class_cost

        # Move the whole cost tensor to CPU once instead of once per image /
        # per slot, so the slot loop below never triggers a device sync.
        total_cost = torch.nan_to_num(torch.clamp(total_cost, -1e10, 1e10), 0)
        total_cost_cpu = total_cost.cpu()

        empty_index = torch.zeros(0, dtype=torch.int64)

        matched_indices = []
        if self.slot_constrained:
            assert query_nums % self.max_prompt_num == 0
            queries_per_slot = query_nums // self.max_prompt_num
            class_targets_cpu = class_targets_padded.cpu()

            for i in range(batch_size):
                n = n_masks_per_image[i]
                if n == 0:
                    matched_indices.append((empty_index, empty_index))
                    continue

                # Which prompt slot each GT of this sample belongs to. A
                # sample may hold fewer targets than max_prompt_num, in which
                # case the remaining slots simply take part in no assignment
                # and all of their queries fall back to the background class.
                per_image_slot_ids = class_targets_cpu[i, :n]

                per_image_pred_index_list = []
                per_image_target_index_list = []
                for per_slot in torch.unique(per_image_slot_ids).tolist():
                    per_slot = int(per_slot)
                    assert 0 <= per_slot < self.max_prompt_num

                    # GT columns owned by this slot: exactly 1 for a 1:1
                    # dataset, M for a 1:M dataset.
                    per_slot_target_index = (
                        per_image_slot_ids == per_slot).nonzero(
                            as_tuple=True)[0]

                    query_start = per_slot * queries_per_slot
                    query_end = query_start + queries_per_slot
                    per_slot_cost_matrix = total_cost_cpu[
                        i, query_start:query_end, :n][:, per_slot_target_index]

                    per_slot_pred_index, per_slot_matched_index = linear_sum_assignment(
                        per_slot_cost_matrix)

                    # Shift the local query index back into global query space.
                    per_image_pred_index_list.append(
                        torch.as_tensor(per_slot_pred_index, dtype=torch.int64)
                        + query_start)
                    per_image_target_index_list.append(
                        per_slot_target_index[torch.as_tensor(
                            per_slot_matched_index, dtype=torch.int64)])

                if len(per_image_pred_index_list) == 0:
                    matched_indices.append((empty_index, empty_index))
                else:
                    matched_indices.append(
                        (torch.cat(per_image_pred_index_list),
                         torch.cat(per_image_target_index_list)))
        else:
            for i in range(batch_size):
                n = n_masks_per_image[i]
                if n == 0:
                    matched_indices.append((empty_index, empty_index))
                    continue

                cost_matrix = total_cost_cpu[i, :, :n]
                per_image_pred_index, per_image_target_index = \
                    linear_sum_assignment(cost_matrix)
                matched_indices.append((torch.as_tensor(per_image_pred_index,
                                                        dtype=torch.int64),
                                        torch.as_tensor(per_image_target_index,
                                                        dtype=torch.int64)))

        return matched_indices


class UniversalSegmentationJointAccelerateLoss(nn.Module):

    def __init__(self,
                 mask_cost=5.0,
                 dice_cost=5.0,
                 class_cost=2.0,
                 mask_loss_weight=5.0,
                 dice_loss_weight=5.0,
                 class_loss_weight=2.0,
                 no_object_class_weight=0.1,
                 vlm_loss_weight=1.0,
                 max_prompt_num=1,
                 slot_constrained_matching=False):
        super(UniversalSegmentationJointAccelerateLoss, self).__init__()
        self.mask_loss_weight = mask_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.class_loss_weight = class_loss_weight
        self.no_object_class_weight = no_object_class_weight
        self.vlm_loss_weight = vlm_loss_weight
        self.max_prompt_num = max_prompt_num
        self.slot_constrained_matching = slot_constrained_matching

        self.hungarian_matcher = Mask2FormerHungarianAccelerateMatcher(
            mask_cost=mask_cost,
            dice_cost=dice_cost,
            class_cost=class_cost,
            max_prompt_num=max_prompt_num,
            slot_constrained=slot_constrained_matching)

        self.sigmoid_ce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def get_pred_permutation_indices(self, indices):
        batch_indices = torch.cat([
            torch.full_like(pred_idx, i)
            for i, (pred_idx, _) in enumerate(indices)
        ])
        pred_indices = torch.cat([pred_idx for (pred_idx, _) in indices])

        return batch_indices, pred_indices

    def get_target_permutation_indices(self, indices):
        batch_indices = torch.cat([
            torch.full_like(target_idx, i)
            for i, (_, target_idx) in enumerate(indices)
        ])
        target_indices = torch.cat([target_idx for (_, target_idx) in indices])

        return batch_indices, target_indices

    def get_assigned_pred_mask_and_target_mask(self, mask_preds, mask_gts,
                                               indices):
        device = mask_preds.device

        pred_idx = self.get_pred_permutation_indices(indices)
        target_idx = self.get_target_permutation_indices(indices)

        pred_masks = mask_preds[pred_idx]

        batch_size, batch_max_object_num, max_height, max_width = len(
            mask_gts), 0, 0, 0
        for per_image_mask_gts in mask_gts:
            object_num, height, width = per_image_mask_gts.shape
            batch_max_object_num = max(batch_max_object_num, object_num)
            max_height = max(max_height, height)
            max_width = max(max_width, width)

        target_masks = torch.zeros(
            [batch_size, batch_max_object_num, max_height, max_width],
            dtype=torch.float32).to(device)
        for idx, per_image_mask_gts in enumerate(mask_gts):
            target_masks[
                idx, :per_image_mask_gts.shape[0], :per_image_mask_gts.
                shape[1], :per_image_mask_gts.shape[2]] = per_image_mask_gts
        target_masks = target_masks[target_idx]

        pred_masks = pred_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        return pred_masks, target_masks

    def compute_batch_mask_loss(self, pred_masks, target_masks):
        mask_loss = self.sigmoid_ce_loss(pred_masks, target_masks)
        mask_loss = mask_loss.mean()

        return mask_loss

    def compute_batch_dice_loss(self, pred_masks, target_masks):
        pred_masks = torch.sigmoid(pred_masks)
        numerator = 2 * (pred_masks * target_masks).sum(dim=-1)
        denominator = pred_masks.sum(dim=-1) + target_masks.sum(dim=-1)
        dice_loss = 1 - (numerator + 1) / (denominator + 1)
        dice_loss = dice_loss.mean()

        return dice_loss

    def compute_batch_class_loss(self, class_preds, class_gts, indices):
        device = class_preds.device
        batch_size, query_nums = class_preds.shape[0], class_preds.shape[1]
        # Dynamic num_classes from class_preds (N_cond + 1 bg)
        # For multi-target visual prompt: class_preds.shape[-1] = N_cond + 1
        # For single-target text prompt: class_preds.shape[-1] = 2
        runtime_num_classes = class_preds.shape[-1]

        idx = self.get_pred_permutation_indices(indices)
        # background class index = runtime_num_classes - 1 (last class)
        class_targets = torch.full((batch_size, query_nums),
                                   fill_value=runtime_num_classes - 1,
                                   dtype=torch.int64,
                                   device=device)

        class_target_objects = torch.cat(
            [target[j] for target, (_, j) in zip(class_gts, indices)])
        class_targets[idx] = class_target_objects

        # Build dynamic CE loss weight: all foreground classes have weight 1.0,
        # background class (last) has weight no_object_class_weight
        ce_weight = torch.ones(runtime_num_classes, device=device)
        ce_weight[-1] = self.no_object_class_weight

        class_preds = class_preds.transpose(1, 2)
        class_loss = F.cross_entropy(class_preds,
                                     class_targets,
                                     weight=ce_weight)

        return class_loss

    def forward(self,
                mask_preds,
                class_preds,
                mask_gts,
                class_gts,
                vlm_loss=None):
        mask_preds = mask_preds.float()
        class_preds = class_preds.float()

        device = mask_preds.device

        mask_gts = [
            per_image_mask_gts.float().to(device)
            for per_image_mask_gts in mask_gts
        ]
        class_gts = [
            per_image_class_gts.long().to(device)
            for per_image_class_gts in class_gts
        ]

        indices = self.hungarian_matcher(mask_preds=mask_preds,
                                         mask_gts=mask_gts,
                                         class_preds=class_preds,
                                         class_gts=class_gts)

        pred_masks, target_masks = self.get_assigned_pred_mask_and_target_mask(
            mask_preds, mask_gts, indices)

        mask_loss = self.compute_batch_mask_loss(pred_masks, target_masks)
        dice_loss = self.compute_batch_dice_loss(pred_masks, target_masks)
        class_loss = self.compute_batch_class_loss(class_preds, class_gts,
                                                   indices)

        mask_loss = self.mask_loss_weight * mask_loss
        dice_loss = self.dice_loss_weight * dice_loss
        class_loss = self.class_loss_weight * class_loss

        if vlm_loss is not None:
            vlm_loss = self.vlm_loss_weight * vlm_loss
        else:
            vlm_loss = torch.tensor(0.0, device=device)

        loss_dict = {
            'vlm_loss': vlm_loss,
            'mask_loss': mask_loss,
            'dice_loss': dice_loss,
            'class_loss': class_loss,
        }

        return loss_dict


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

    from tools.path import interactive_segmentation_dataset_path, text_prompt_segmentation_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimplerPromptSegmentation.datasets.prompt_segmentation_dataset import PromptSegmentationDataset
    from SimplerPromptSegmentation.prompt_segmentation_common import PromptSegmentationResize, Normalize, PromptSegmentationTrainCollater

    prompt_segmentation_dataset = PromptSegmentationDataset(
        #########################################################
        visual_prompt_image_root_dir=interactive_segmentation_dataset_path,
        visual_prompt_image_set_name=[
            'sa_000000',
        ],
        visual_prompt_image_set_type='train',
        visual_prompt_image_per_set_image_choose_max_num={
            'sa_000000': 1000000,
        },
        visual_prompt_per_image_mask_choose_max_num=32,
        visual_prompt_per_image_sample_num=4,
        visual_prompt_area_filter_ratio=0.001,
        visual_prompt_box_noise_wh_ratio=0.1,
        visual_prompt_mask_noise_area_ratio=0.04,
        #########################################################
        text_prompt_image_root_dir=text_prompt_segmentation_dataset_path,
        text_prompt_image_set_name=[
            'sa1b_0_0',
        ],
        text_prompt_image_set_type='train',
        text_prompt_image_per_set_image_choose_max_num={
            'sa1b_0_0': 1000000,
        },
        text_prompt_per_image_mask_choose_max_num=32,
        text_prompt_per_image_sample_num=4,
        text_prompt_area_filter_ratio=0.001,
        #########################################################
        transform=transforms.Compose([
            PromptSegmentationResize(resize=1024,
                                     stride=32,
                                     multi_scale=False,
                                     multi_scale_range=[0.8, 1.0]),
            Normalize(mean=[123.675, 116.28, 103.53],
                      std=[58.395, 57.12, 57.375]),
        ]))

    from torch.utils.data import DataLoader
    collater = PromptSegmentationTrainCollater(
        resize=1024,
        pil_resize=1024,
        use_prompt_sample_type_prob={
            'visual': 0.5,
            'text': 0.5,
        },
        use_prompt_language_type_prob={
            'chinese': 0.5,
            'english': 0.5,
        },
        use_visual_prompt_type_prob={
            'prompt_point': 0.35,
            'prompt_box': 0.35,
            'prompt_mask': 0.3,
        },
        use_visual_prompt_num_prob={
            1: 0.3,
            2: 0.3,
            3: 0.2,
            4: 0.2,
        },
        use_text_prompt_num_prob={
            1: 0.25,
            2: 0.25,
            3: 0.25,
            4: 0.25,
        },
        use_description_type_prob={
            'absolute_position': 0.,
            'category': 0.,
            'phrase_description': 0.25,
            'detail_description': 0.25,
            'absolute_detail_description': 0.25,
            'relative_detail_description': 0.25,
        },
        point_radius=10,
        visual_prompt_type_to_id={
            'prompt_point': 0,
            'prompt_box': 1,
            'prompt_mask': 2,
        },
        seed=0,
    )
    train_loader = DataLoader(prompt_segmentation_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=collater)

    from SimplerPromptSegmentation.models.tokenizer import Qwen3VLSegTokenizer
    from SimplerPromptSegmentation.models.qwen3vl_dinov3_prompt_segmentation import qwen3vl_dinov3_vit_base_patch16_prompt_segmentation
    vlm_model_path = "Qwen/Qwen3-VL-4B-Instruct"
    tokenizer = Qwen3VLSegTokenizer(vlm_model_path)
    net = qwen3vl_dinov3_vit_base_patch16_prompt_segmentation(
        vlm_model_path=vlm_model_path,
        tokenizer_vocab_size=tokenizer.vocab_size,
        max_prompt_num=4,
        use_gradient_checkpoint=True)
    net = net.cuda()

    loss1 = UniversalSegmentationJointAccelerateLoss(
        mask_cost=5.0,
        dice_cost=5.0,
        class_cost=2.0,
        mask_loss_weight=5.0,
        dice_loss_weight=5.0,
        class_loss_weight=2.0,
        no_object_class_weight=0.1,
        vlm_loss_weight=1.0,
        max_prompt_num=4)
    for data in tqdm(train_loader):
        images, masks, pil_images = data['image'], data['mask'], data[
            'pil_image']
        prompt_texts, prompt_languages = data['prompt_text'], data[
            'prompt_language']
        sample_type = data['sample_type']

        images = images.cuda()
        mask_gts = [m.cuda() for m in masks]
        class_gts = [
            torch.arange(m.shape[0], dtype=torch.long).cuda() for m in masks
        ]

        print('1111', images.shape, len(masks), len(class_gts), sample_type)

        vprompt_masks = None
        if sample_type == 'visual':
            vprompt_masks = [m.cuda() for m in data['prompt_mask']]

        for per_image_masks, per_image_labels in zip(masks, class_gts):
            print('2222', per_image_masks.shape)
            print('3333', len(per_image_labels), per_image_labels)

        tokenized = tokenizer.encode(prompt_texts=prompt_texts,
                                     sample_type=sample_type,
                                     prompt_languages=prompt_languages,
                                     pil_images=pil_images)

        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()
        labels = tokenized['labels'].cuda()
        cond_ids = tokenized['cond_ids'].cuda()
        seg_ids = tokenized['seg_ids'].cuda()
        pixel_values = tokenized['pixel_values'].cuda()
        image_grid_thw = tokenized['image_grid_thw'].cuda()
        mm_token_type_ids = tokenized['mm_token_type_ids'].cuda()

        region_token_id = tokenizer.region_token_id

        mask_preds, class_preds, vlm_loss = net(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            cond_ids=cond_ids,
            seg_ids=seg_ids,
            vprompt_masks=vprompt_masks,
            labels=labels,
            region_token_id=region_token_id)

        print('4444', mask_preds.shape, class_preds.shape, vlm_loss)

        out = loss1(mask_preds,
                    class_preds,
                    mask_gts,
                    class_gts,
                    vlm_loss=vlm_loss)
        print('5555', out)

        break
