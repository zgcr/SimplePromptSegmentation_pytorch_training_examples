import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'PromptSegmentationDecoder',
]


class PromptSegmentationDecoder(nn.Module):

    def __init__(self,
                 topk=100,
                 min_score_threshold=0.1,
                 mask_threshold=0.5,
                 binary_mask=True,
                 per_prompt_mode='argmax',
                 mask_nms_iou_threshold=0.7,
                 max_prompt_num=1):
        super(PromptSegmentationDecoder, self).__init__()
        self.topk = topk
        self.min_score_threshold = min_score_threshold
        self.mask_threshold = mask_threshold
        self.binary_mask = binary_mask
        self.per_prompt_mode = per_prompt_mode
        self.mask_nms_iou_threshold = mask_nms_iou_threshold
        self.max_prompt_num = max_prompt_num

        # 'argmax': one mask per prompt   (1 prompt : 1 mask)
        # 'thresh': every query confident enough for this prompt, deduplicated
        #           by mask-NMS  (1 prompt : M masks, M is adaptive)
        assert per_prompt_mode in ['argmax', 'thresh']
        assert min_score_threshold > 0, 'min_score_threshold must be > 0'

    def mask_nms(self, masks, scores):
        # binarize once for IoU computation
        binary_masks = (masks > self.mask_threshold).flatten(1).float()
        areas = binary_masks.sum(dim=1)

        intersection = binary_masks @ binary_masks.t()
        union = areas[:, None] + areas[None, :] - intersection
        ious = intersection / union.clamp(min=1e-6)

        keep_indices = []
        for i in range(binary_masks.shape[0]):
            duplicated = False
            for j in keep_indices:
                if ious[i, j] > self.mask_nms_iou_threshold:
                    duplicated = True
                    break
            if not duplicated:
                keep_indices.append(i)

        keep_indices = torch.as_tensor(keep_indices,
                                       dtype=torch.long,
                                       device=masks.device)

        return masks[keep_indices], scores[keep_indices]

    def select_queries_for_prompt(self, per_prompt_scores):
        if self.per_prompt_mode == 'argmax':
            best_score, best_query = torch.max(per_prompt_scores, dim=0)
            selected_scores = best_score.reshape(1)
            selected_queries = best_query.reshape(1)
        else:
            selected_queries = (per_prompt_scores
                                > self.min_score_threshold).nonzero(
                                    as_tuple=True)[0]
            selected_scores = per_prompt_scores[selected_queries]
            sort_indexs = torch.argsort(selected_scores, descending=True)
            selected_queries = selected_queries[sort_indexs]
            selected_scores = selected_scores[sort_indexs]

        # apply the score threshold for all modes so that a prompt the model is
        # not confident about produces no mask instead of a random one
        keep_flag = selected_scores > self.min_score_threshold
        selected_queries = selected_queries[keep_flag]
        selected_scores = selected_scores[keep_flag]

        return selected_queries, selected_scores

    def forward(self, preds, scaled_sizes, origin_sizes):
        with torch.no_grad():
            mask_preds, class_preds = preds

            query_num = mask_preds.shape[1]

            mask_preds = torch.sigmoid(mask_preds)
            class_preds = torch.softmax(class_preds, dim=-1)

            # remove background class (last class index), the remaining
            # dimension indexes the prompts
            class_preds = class_preds[:, :, :-1]
            per_image_prompt_nums = class_preds.shape[-1]

            # Slot constrained decoding, mirroring the slot constrained
            # Hungarian matching used at training time: the model binds query
            # q to prompt slot q // queries_per_slot, so prompt k may only be
            # served by the queries that were conditioned on prompt k.
            assert query_num % self.max_prompt_num == 0
            assert per_image_prompt_nums <= self.max_prompt_num
            queries_per_slot = query_num // self.max_prompt_num

            batch_masks, batch_scores, batch_classes = [], [], []
            for per_image_idx, (per_image_mask_preds, per_image_class_preds,
                                per_image_scaled_sizes,
                                per_image_origin_sizes) in enumerate(
                                    zip(mask_preds, class_preds, scaled_sizes,
                                        origin_sizes)):

                per_image_origin_h, per_image_origin_w = int(
                    per_image_origin_sizes[0]), int(per_image_origin_sizes[1])

                # Empty results must use the same resolution as the non-empty
                # ones (the original image size), otherwise downstream code
                # that assumes a consistent shape breaks.
                empty_per_image_mask_preds = np.zeros(
                    (0, per_image_origin_h, per_image_origin_w),
                    dtype=np.float32)
                empty_per_image_pred_scores = np.zeros((0), dtype=np.float32)
                empty_per_image_pred_classes = np.zeros((0), dtype=np.float32)

                per_image_keep_mask_preds = []
                per_image_keep_pred_scores = []
                per_image_keep_pred_classes = []
                for per_prompt_idx in range(per_image_prompt_nums):
                    # Prompt k may only pick from the queries the model
                    # conditioned on prompt k. Both 'argmax' and 'thresh'
                    # therefore rank queries within a single slot:
                    #   argmax -> the best query of the slot (0 or 1 mask)
                    #   thresh -> every confident query of the slot, later
                    #             deduplicated by mask-NMS (0..N masks)
                    query_offset = per_prompt_idx * queries_per_slot
                    # [queries_per_slot] score of this slot's queries
                    per_prompt_scores = per_image_class_preds[
                        query_offset:query_offset + queries_per_slot,
                        per_prompt_idx]

                    selected_queries, selected_scores = self.select_queries_for_prompt(
                        per_prompt_scores)

                    if selected_queries.shape[0] == 0:
                        continue

                    # map the slot local query index back to the global one
                    selected_queries = selected_queries + query_offset

                    per_prompt_mask_preds = per_image_mask_preds[
                        selected_queries]

                    # multiple queries per prompt usually collapse onto the same
                    # instance, so remove the duplicates
                    if self.per_prompt_mode != 'argmax' and per_prompt_mask_preds.shape[
                            0] > 1:
                        per_prompt_mask_preds, selected_scores = self.mask_nms(
                            per_prompt_mask_preds, selected_scores)

                    per_image_keep_mask_preds.append(per_prompt_mask_preds)
                    per_image_keep_pred_scores.append(selected_scores)
                    per_image_keep_pred_classes.append(
                        torch.full((per_prompt_mask_preds.shape[0], ),
                                   per_prompt_idx,
                                   dtype=torch.long,
                                   device=per_prompt_mask_preds.device))

                if len(per_image_keep_mask_preds) == 0:
                    batch_masks.append(empty_per_image_mask_preds)
                    batch_scores.append(empty_per_image_pred_scores)
                    batch_classes.append(empty_per_image_pred_classes)
                    continue

                per_image_mask_preds = torch.cat(per_image_keep_mask_preds,
                                                 dim=0)
                per_image_pred_scores = torch.cat(per_image_keep_pred_scores,
                                                  dim=0)
                per_image_pred_classes = torch.cat(per_image_keep_pred_classes,
                                                   dim=0)

                # final safety cap on the number of masks per image. The
                # per-prompt grouping above is preserved as much as possible:
                # only the globally lowest scoring masks are dropped.
                if per_image_pred_scores.shape[0] > self.topk:
                    sort_indexs = torch.argsort(per_image_pred_scores,
                                                descending=True)[:self.topk]
                    sort_indexs, _ = torch.sort(sort_indexs)
                    per_image_mask_preds = per_image_mask_preds[sort_indexs]
                    per_image_pred_scores = per_image_pred_scores[sort_indexs]
                    per_image_pred_classes = per_image_pred_classes[
                        sort_indexs]

                per_image_mask_preds = per_image_mask_preds[:, :int(
                    per_image_scaled_sizes[0]), :int(per_image_scaled_sizes[1]
                                                     )]

                per_image_mask_preds = F.interpolate(
                    per_image_mask_preds.unsqueeze(0),
                    size=[per_image_origin_h, per_image_origin_w],
                    mode='bilinear')
                per_image_mask_preds = per_image_mask_preds.squeeze(0)

                if self.binary_mask:
                    per_image_mask_preds = (per_image_mask_preds
                                            > self.mask_threshold).to(
                                                torch.uint8)

                per_image_mask_preds = per_image_mask_preds.cpu().numpy()
                per_image_pred_scores = per_image_pred_scores.cpu().numpy()
                per_image_pred_classes = per_image_pred_classes.cpu().numpy()

                batch_masks.append(per_image_mask_preds)
                batch_scores.append(per_image_pred_scores)
                batch_classes.append(per_image_pred_classes)

            return batch_masks, batch_scores, batch_classes


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
    from SimplerPromptSegmentation.prompt_segmentation_common import PromptSegmentationResize, Normalize, PromptSegmentationTestCollater

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
    collater = PromptSegmentationTestCollater(
        resize=1024,
        pil_resize=1024,
        use_prompt_sample_type='visual',
        use_prompt_language_type='english',
        use_visual_prompt_type='prompt_box',
        use_description_type='detail_description',
        use_visual_prompt_num=1,
        use_text_prompt_num=1,
        point_radius=10,
        visual_prompt_type_to_id={
            'prompt_point': 0,
            'prompt_box': 1,
            'prompt_mask': 2,
        },
    )
    train_loader = DataLoader(prompt_segmentation_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from SimplerPromptSegmentation.models.tokenizer import Qwen3VLSegTokenizer
    from SimplerPromptSegmentation.models.qwen3vl_dinov3_prompt_segmentation import qwen3vl_dinov3_vit_base_patch16_prompt_segmentation
    vlm_model_path = "Qwen/Qwen3-VL-4B-Instruct"
    tokenizer = Qwen3VLSegTokenizer(vlm_model_path)
    net = qwen3vl_dinov3_vit_base_patch16_prompt_segmentation(
        vlm_model_path=vlm_model_path,
        tokenizer_vocab_size=tokenizer.vocab_size,
        max_prompt_num=4)
    net = net.cuda()
    net.eval()

    # 'argmax': exactly one mask per prompt (1 prompt : 1 mask tasks)
    decode = PromptSegmentationDecoder(topk=100,
                                       min_score_threshold=0.1,
                                       mask_threshold=0.5,
                                       binary_mask=True,
                                       per_prompt_mode='argmax',
                                       mask_nms_iou_threshold=0.7,
                                       max_prompt_num=4)
    # 'thresh': as many masks per prompt as the model is confident about
    # (1 prompt : M masks), deduplicated by mask-NMS. e.g. one "person"
    # prompt -> every person instance.
    thresh_decode = PromptSegmentationDecoder(topk=100,
                                              min_score_threshold=0.1,
                                              mask_threshold=0.5,
                                              binary_mask=True,
                                              per_prompt_mode='thresh',
                                              mask_nms_iou_threshold=0.7,
                                              max_prompt_num=4)

    for data in tqdm(train_loader):
        images, masks, sizes, origin_sizes = data['image'], data['mask'], data[
            'size'], data['origin_size']

        print('1111', images.shape, len(masks), sizes.shape,
              origin_sizes.shape)

        pil_images, prompt_texts, prompt_languages, sample_type = data[
            'pil_image'], data['prompt_text'], data['prompt_language'], data[
                'sample_type']

        tokenized = tokenizer.encode(prompt_texts=prompt_texts,
                                     sample_type=sample_type,
                                     prompt_languages=prompt_languages,
                                     pil_images=pil_images)
        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()
        pixel_values = tokenized['pixel_values'].cuda()
        image_grid_thw = tokenized['image_grid_thw'].cuda()
        mm_token_type_ids = tokenized['mm_token_type_ids'].cuda()
        cond_ids = tokenized['cond_ids'].cuda()
        seg_ids = tokenized['seg_ids'].cuda()
        labels = tokenized['labels'].cuda()

        vprompt_masks = None
        prompt_type_id = None
        if sample_type == 'visual':
            vprompt_masks = [m.cuda() for m in data['prompt_mask']]
            if data.get('visual_prompt_type_id', None) is not None:
                prompt_type_id = data['visual_prompt_type_id'].cuda()

        valid_sizes = torch.as_tensor(sizes).float().cuda()

        with torch.no_grad():
            mask_preds, class_preds, vlm_loss = net(
                images=images.cuda(),
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                cond_ids=cond_ids,
                seg_ids=seg_ids,
                vprompt_masks=vprompt_masks,
                labels=labels,
                region_token_id=tokenizer.region_token_id,
                valid_sizes=valid_sizes,
                prompt_type_id=prompt_type_id,
            )

        print('2222', mask_preds.shape, class_preds.shape, vlm_loss)

        # the decoder derives the prompt count from class_preds itself, so
        # num_targets is only printed here for reference
        prompt_nums = torch.as_tensor(data['num_targets'])

        for per_mode_name, per_mode_decode in [('argmax', decode),
                                               ('thresh', thresh_decode)]:
            batch_masks, batch_scores, batch_classes = per_mode_decode(
                [mask_preds, class_preds], sizes, origin_sizes)

            print('3333', f'per_prompt_mode: {per_mode_name}',
                  f'prompt_nums: {prompt_nums.tolist()}')

            for per_image_mask_preds, per_image_pred_scores, per_image_pred_classes in zip(
                    batch_masks, batch_scores, batch_classes):
                print('4444', per_image_mask_preds.shape,
                      per_image_pred_scores.shape,
                      per_image_pred_classes.shape,
                      f'prompt_ids: {per_image_pred_classes.tolist()}')
        break
