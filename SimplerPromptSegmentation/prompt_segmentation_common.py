import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F


class PromptSegmentationResize:

    def __init__(self,
                 resize=1024,
                 stride=32,
                 multi_scale=False,
                 multi_scale_range=[0.8, 1.0]):

        self.resize = resize
        self.stride = stride
        self.multi_scale = multi_scale
        self.multi_scale_range = multi_scale_range

        assert 0.0 < self.multi_scale_range[0] <= 1.0
        assert 0.0 < self.multi_scale_range[1] <= 1.0
        assert self.multi_scale_range[0] <= self.multi_scale_range[1]

    def resize_mask(self, mask, resize_w, resize_h):
        mask = cv2.resize(mask, (resize_w, resize_h),
                          interpolation=cv2.INTER_NEAREST)

        return mask

    def __call__(self, sample):
        image, pil_image, size = sample['image'], sample['pil_image'], sample[
            'size']

        h, w, _ = image.shape

        if self.multi_scale:
            scale_range = [
                int(self.multi_scale_range[0] * self.resize),
                int(self.multi_scale_range[1] * self.resize)
            ]
            resize_list = [
                i // self.stride * self.stride
                for i in range(scale_range[0], scale_range[1] + self.stride)
            ]
            resize_list = list(set(resize_list))

            random_idx = np.random.randint(0, len(resize_list))
            final_resize = resize_list[random_idx]
        else:
            final_resize = self.resize

        factor = final_resize / max(h, w)

        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(image, (resize_w, resize_h))
        pil_image = pil_image.resize((resize_w, resize_h))

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample['image'], sample['pil_image'], sample[
            'size'] = image, pil_image, size

        if sample['sample_type'] == 'visual':
            # Resize multiple GT masks and prompt masks
            masks = sample['masks']
            prompt_masks = sample['prompt_masks']

            with ThreadPoolExecutor(max_workers=4) as executor:
                resized_masks = list(
                    executor.map(
                        lambda mask: self.resize_mask(mask, resize_w, resize_h
                                                      ), masks))
                resized_prompt_masks = list(
                    executor.map(
                        lambda mask: self.resize_mask(mask, resize_w, resize_h
                                                      ), prompt_masks))

            sample['masks'] = resized_masks
            sample['prompt_masks'] = resized_prompt_masks

            # Resize prompt points: scale (x, y) coordinates
            prompt_points = sample['prompt_points']
            resized_prompt_points = []
            for pts in prompt_points:
                pts = pts.copy()
                pts[:, 0] = pts[:, 0] * (float(resize_w) / w)
                pts[:, 1] = pts[:, 1] * (float(resize_h) / h)
                resized_prompt_points.append(pts)
            sample['prompt_points'] = resized_prompt_points

            # Resize prompt boxes: scale (x_min, y_min, x_max, y_max)
            prompt_boxes = sample['prompt_boxes']
            resized_prompt_boxes = []
            for boxes in prompt_boxes:
                boxes = boxes.copy()
                boxes[:, 0] = boxes[:, 0] * (float(resize_w) / w)
                boxes[:, 1] = boxes[:, 1] * (float(resize_h) / h)
                boxes[:, 2] = boxes[:, 2] * (float(resize_w) / w)
                boxes[:, 3] = boxes[:, 3] * (float(resize_h) / h)
                resized_prompt_boxes.append(boxes)
            sample['prompt_boxes'] = resized_prompt_boxes

        elif sample['sample_type'] == 'text':
            # Resize multiple GT masks (multi-target text prompt)
            masks = sample['masks']

            with ThreadPoolExecutor(max_workers=4) as executor:
                resized_masks = list(
                    executor.map(
                        lambda mask: self.resize_mask(mask, resize_w, resize_h
                                                      ), masks))
            sample['masks'] = resized_masks

        return sample


class Normalize:

    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]):
        self.mean = np.expand_dims(np.expand_dims(np.array(mean), axis=0),
                                   axis=0)
        self.std = np.expand_dims(np.expand_dims(np.array(std), axis=0),
                                  axis=0)

    def __call__(self, sample):
        image = sample['image']

        image = (image - self.mean) / self.std

        sample['image'] = image

        return sample


class PromptSegmentationTrainCollater:

    def __init__(
        self,
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
    ):
        self.resize = resize
        self.pil_resize = pil_resize
        self.use_prompt_sample_type_prob = use_prompt_sample_type_prob

        self.use_prompt_language_type_prob = use_prompt_language_type_prob
        self.use_visual_prompt_type_prob = use_visual_prompt_type_prob
        self.use_visual_prompt_num_prob = use_visual_prompt_num_prob
        self.use_text_prompt_num_prob = use_text_prompt_num_prob
        self.use_description_type_prob = use_description_type_prob
        self.point_radius = point_radius
        self.visual_prompt_type_to_id = visual_prompt_type_to_id

        assert resize % 64 == 0
        assert self.pil_resize % 32 == 0

        assert abs(
            sum(self.use_prompt_sample_type_prob.values()) -
            1.) < 1e-6, 'use_prompt_sample_type_prob values must sum to 1'
        assert abs(
            sum(self.use_prompt_language_type_prob.values()) -
            1.) < 1e-6, 'use_prompt_language_type_prob values must sum to 1'
        assert abs(
            sum(self.use_visual_prompt_type_prob.values()) -
            1.) < 1e-6, 'use_visual_prompt_type_prob values must sum to 1'
        assert abs(
            sum(self.use_visual_prompt_num_prob.values()) -
            1.) < 1e-6, 'use_visual_prompt_num_prob values must sum to 1'
        assert abs(sum(self.use_text_prompt_num_prob.values()) -
                   1.) < 1e-6, 'use_text_prompt_num_prob values must sum to 1'
        assert abs(sum(self.use_description_type_prob.values()) -
                   1.) < 1e-6, 'use_description_type_prob values must sum to 1'

        self.seed = seed
        self.epoch = 0
        self.local_batch_count = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.local_batch_count = 0

    def get_batch_random_state(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        batch_index = self.local_batch_count * num_workers + worker_id
        self.local_batch_count += 1

        return np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, batch_index]))

    def random_choice_from_prob_dict(self, prob_dict, random_state=None):
        """根据概率字典随机选择一个key

        random_state is the per-batch synchronized RNG. It is None only for
        the standalone __main__ demos, which fall back to the global np.random.
        """
        keys = list(prob_dict.keys())
        probs = list(prob_dict.values())
        if random_state is None:
            return keys[int(np.random.choice(len(keys), p=probs))]

        return keys[int(random_state.choice(len(keys), p=probs))]

    def __call__(self, data):
        # Every batch level choice below is drawn from this RNG, which depends
        # only on (seed, epoch, batch index) and is therefore identical on
        # every rank at the same step.
        batch_random_state = self.get_batch_random_state()

        # ---- Select sample type (visual/text) for this batch ----
        selected_sample_type = str(
            self.random_choice_from_prob_dict(self.use_prompt_sample_type_prob,
                                              batch_random_state))

        if selected_sample_type == 'visual':
            data = [s['visual_prompt_sample'] for s in data]
        else:
            data = [s['text_prompt_sample'] for s in data]

        sample_type = data[0]['sample_type']

        images = [s['image'] for s in data]
        sizes = [s['size'] for s in data]
        # origin_size is the pre-transform (height, width) of each image. It is
        # produced by the dataset and never touched by PromptSegmentationResize,
        # unlike `size` which holds the resized (letterbox) size.
        origin_sizes = [s['origin_size'] for s in data]
        pil_images = [s['pil_image'] for s in data]

        input_pil_images = []
        for per_pil_image in pil_images:
            w, h = per_pil_image.size
            factor = self.pil_resize / max(h, w)
            resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
            per_pil_image = per_pil_image.resize((resize_w, resize_h))
            input_pil_images.append(per_pil_image)

        input_images = []
        for i, per_image in enumerate(images):
            per_input_image = np.zeros((self.resize, self.resize, 3),
                                       dtype=np.float32)
            per_input_image[0:per_image.shape[0],
                            0:per_image.shape[1], :] = per_image
            per_input_image = torch.from_numpy(per_input_image)
            # [3,H,W]
            per_input_image = per_input_image.permute(2, 0, 1)
            input_images.append(per_input_image)
        input_images = torch.stack(input_images, dim=0)
        input_images = input_images.float()

        sizes = np.array(sizes, dtype=np.float32)
        origin_sizes = np.array(origin_sizes, dtype=np.float32)

        result = {
            'image': input_images,
            'size': sizes,
            'origin_size': origin_sizes,
            'sample_type': sample_type,
            'pil_image': input_pil_images,
        }

        if sample_type == 'visual':
            # ---- Batch-level visual prompt num sampling ----
            batch_target_num = int(
                self.random_choice_from_prob_dict(
                    self.use_visual_prompt_num_prob, batch_random_state))
            for s in data:
                available = len(s['masks'])

                actual_num = min(batch_target_num, available)
                if actual_num < available:
                    indices = np.random.choice(available,
                                               actual_num,
                                               replace=False).tolist()
                    s['masks'] = [s['masks'][i] for i in indices]
                    s['prompt_points'] = [
                        s['prompt_points'][i] for i in indices
                    ]
                    s['prompt_boxes'] = [s['prompt_boxes'][i] for i in indices]
                    s['prompt_masks'] = [s['prompt_masks'][i] for i in indices]
                s['num_targets'] = len(s['masks'])

            # ---- Multi-target visual prompt ----
            all_masks = [s['masks'] for s in data]
            num_targets_list = [s['num_targets'] for s in data]

            # Pad GT masks: list of [N_i, H, W] tensors
            input_masks_list = []
            for per_sample_masks in all_masks:
                per_sample_input_masks = []
                for per_mask in per_sample_masks:
                    per_input_mask = np.zeros((self.resize, self.resize),
                                              dtype=np.float32)
                    per_input_mask[0:per_mask.shape[0],
                                   0:per_mask.shape[1]] = per_mask
                    per_input_mask = torch.from_numpy(per_input_mask)
                    per_sample_input_masks.append(per_input_mask)
                per_sample_input_masks = torch.stack(per_sample_input_masks,
                                                     dim=0)
                input_masks_list.append(per_sample_input_masks)

            result['mask'] = input_masks_list  # list of [N_i, H, W]
            result['num_targets'] = num_targets_list

            # ---- Build instance_to_prompt_ids ----
            instance_to_prompt_ids_list = []
            for n in num_targets_list:
                instance_to_prompt_ids_list.append(
                    torch.arange(n, dtype=torch.long))
            result['instance_to_prompt_ids'] = instance_to_prompt_ids_list

            # ---- Select visual prompt type for this batch ----
            selected_visual_prompt_type = str(
                self.random_choice_from_prob_dict(
                    self.use_visual_prompt_type_prob, batch_random_state))

            # ---- Convert selected prompt type to unified mask ----
            input_prompt_masks_list = []

            if selected_visual_prompt_type == 'prompt_point':
                # Point → mask: mark point pixels, enhance with circles
                all_prompt_points = [s['prompt_points'] for s in data]
                point_kernel_size = 2 * self.point_radius + 1
                point_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (point_kernel_size, point_kernel_size))

                for sample_idx, per_sample_prompt_points in enumerate(
                        all_prompt_points):
                    per_sample_input_prompt_masks = []
                    for per_prompt_point in per_sample_prompt_points:
                        # Create mask at self.resize resolution
                        point_mask = np.zeros((self.resize, self.resize),
                                              dtype=np.uint8)
                        for pt_idx in range(per_prompt_point.shape[0]):
                            px = int(
                                np.clip(per_prompt_point[pt_idx, 0], 0,
                                        self.resize - 1))
                            py = int(
                                np.clip(per_prompt_point[pt_idx, 1], 0,
                                        self.resize - 1))
                            point_mask[py, px] = 1
                        # Enhance with circles (dilate)
                        point_mask = cv2.dilate(point_mask,
                                                point_kernel,
                                                iterations=1)
                        point_mask = point_mask.astype(np.float32)
                        per_sample_input_prompt_masks.append(
                            torch.from_numpy(point_mask))
                    per_sample_input_prompt_masks = torch.stack(
                        per_sample_input_prompt_masks, dim=0)
                    input_prompt_masks_list.append(
                        per_sample_input_prompt_masks)

            elif selected_visual_prompt_type == 'prompt_box':
                # Box → mask: fill box regions with 1
                all_prompt_boxes = [s['prompt_boxes'] for s in data]

                for sample_idx, per_sample_prompt_boxes in enumerate(
                        all_prompt_boxes):
                    per_sample_input_prompt_masks = []
                    for per_target_prompt_boxes in per_sample_prompt_boxes:
                        # Create mask at self.resize resolution
                        box_mask = np.zeros((self.resize, self.resize),
                                            dtype=np.float32)
                        for per_prompt_box in per_target_prompt_boxes:
                            x_min = int(
                                np.clip(per_prompt_box[0], 0, self.resize - 1))
                            y_min = int(
                                np.clip(per_prompt_box[1], 0, self.resize - 1))
                            x_max = int(
                                np.clip(per_prompt_box[2], 0, self.resize - 1))
                            y_max = int(
                                np.clip(per_prompt_box[3], 0, self.resize - 1))
                            box_mask[y_min:y_max + 1, x_min:x_max + 1] = 1
                        per_sample_input_prompt_masks.append(
                            torch.from_numpy(box_mask))
                    per_sample_input_prompt_masks = torch.stack(
                        per_sample_input_prompt_masks, dim=0)
                    input_prompt_masks_list.append(
                        per_sample_input_prompt_masks)

            elif selected_visual_prompt_type == 'prompt_mask':
                # Mask → mask: use existing noised prompt masks, pad to self.resize
                all_prompt_masks = [s['prompt_masks'] for s in data]

                for per_sample_prompt_masks in all_prompt_masks:
                    per_sample_input_prompt_masks = []
                    for per_prompt_mask in per_sample_prompt_masks:
                        padded = np.zeros((self.resize, self.resize),
                                          dtype=np.float32)
                        padded[0:per_prompt_mask.shape[0],
                               0:per_prompt_mask.shape[1]] = per_prompt_mask
                        per_sample_input_prompt_masks.append(
                            torch.from_numpy(padded))
                    per_sample_input_prompt_masks = torch.stack(
                        per_sample_input_prompt_masks, dim=0)
                    input_prompt_masks_list.append(
                        per_sample_input_prompt_masks)

            # list of [N_i, resize, resize]
            result['prompt_mask'] = input_prompt_masks_list
            # Export the prompt type so the model can distinguish a coarse
            # box prompt from an accurate mask prompt.
            result['visual_prompt_type'] = selected_visual_prompt_type
            result['visual_prompt_type_id'] = torch.full(
                (len(data), ),
                self.visual_prompt_type_to_id[selected_visual_prompt_type],
                dtype=torch.long)

            # ---- Raw text prompts for visual prompt (multi-region) ----
            # Each sample gets N_i <region> tokens joined by ", "
            prompt_texts = []
            for n in num_targets_list:
                prompt_texts.append(', '.join(['<region>'] * n))

            # ---- Select language type for the batch ----
            selected_language = str(
                self.random_choice_from_prob_dict(
                    self.use_prompt_language_type_prob, batch_random_state))
            prompt_languages = [selected_language] * len(data)

            result['prompt_text'] = prompt_texts
            result['prompt_language'] = prompt_languages

        else:
            result['visual_prompt_type'] = None
            result['visual_prompt_type_id'] = None

            # ---- Batch-level text prompt num sampling ----
            batch_target_num = int(
                self.random_choice_from_prob_dict(
                    self.use_text_prompt_num_prob, batch_random_state))
            for s in data:
                available = len(s['masks'])

                actual_num = min(batch_target_num, available)
                if actual_num < available:
                    indices = np.random.choice(available,
                                               actual_num,
                                               replace=False).tolist()
                    s['masks'] = [s['masks'][i] for i in indices]
                    s['prompt_texts'] = [s['prompt_texts'][i] for i in indices]
                s['num_targets'] = len(s['masks'])

            # ---- Multi-target text prompt ----
            # Each sample has a list of GT masks and a list of text descriptions
            # list of list of [H,W]
            all_masks = [s['masks'] for s in data]
            # list of list of desc_dict
            all_text_prompts = [s['prompt_texts'] for s in data]
            num_targets_list = [s['num_targets'] for s in data]

            # Pad GT masks: list of [N_i, H, W] tensors
            input_masks_list = []
            for per_sample_masks in all_masks:
                per_sample_input_masks = []
                for per_mask in per_sample_masks:
                    per_input_mask = np.zeros((self.resize, self.resize),
                                              dtype=np.float32)
                    per_input_mask[0:per_mask.shape[0],
                                   0:per_mask.shape[1]] = per_mask
                    per_input_mask = torch.from_numpy(per_input_mask)
                    per_sample_input_masks.append(per_input_mask)
                per_sample_input_masks = torch.stack(per_sample_input_masks,
                                                     dim=0)
                input_masks_list.append(per_sample_input_masks)

            # list of [N_i, H, W]
            result['mask'] = input_masks_list
            result['num_targets'] = num_targets_list

            # ---- Build instance_to_prompt_ids ----
            instance_to_prompt_ids_list = []
            for n in num_targets_list:
                instance_to_prompt_ids_list.append(
                    torch.arange(n, dtype=torch.long))
            result['instance_to_prompt_ids'] = instance_to_prompt_ids_list

            # ---- Select language type for the batch ----
            batch_language = str(
                self.random_choice_from_prob_dict(
                    self.use_prompt_language_type_prob, batch_random_state))

            # Select description type for this batch (consistent across batch)
            selected_description_key = str(
                self.random_choice_from_prob_dict(
                    self.use_description_type_prob, batch_random_state))

            # list of list of str
            origin_text_prompts = []
            prompt_languages = []
            for per_sample_text_prompts in all_text_prompts:
                per_sample_labels = []
                for per_mask_desc_dict in per_sample_text_prompts:
                    language_text_dict = per_mask_desc_dict[batch_language]
                    selected_text = language_text_dict[
                        selected_description_key]
                    label = selected_text.strip().lower()
                    per_sample_labels.append(label)
                origin_text_prompts.append(per_sample_labels)
                prompt_languages.append(batch_language)

            result['prompt_text'] = origin_text_prompts
            result['prompt_language'] = prompt_languages

        return result


class PromptSegmentationTestCollater:

    def __init__(
        self,
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
    ):
        self.resize = resize
        self.pil_resize = pil_resize
        self.use_prompt_sample_type = use_prompt_sample_type
        self.use_prompt_language_type = use_prompt_language_type
        self.use_visual_prompt_type = use_visual_prompt_type
        self.use_description_type = use_description_type
        self.use_visual_prompt_num = use_visual_prompt_num
        self.use_text_prompt_num = use_text_prompt_num
        self.point_radius = point_radius
        self.visual_prompt_type_to_id = visual_prompt_type_to_id

        assert resize % 64 == 0
        assert self.pil_resize % 32 == 0

        assert use_prompt_sample_type in ['visual', 'text']
        assert use_prompt_language_type in ['chinese', 'english']
        assert use_visual_prompt_type in [
            'prompt_point', 'prompt_box', 'prompt_mask'
        ]
        assert use_description_type in [
            'absolute_position', 'category', 'phrase_description',
            'detail_description', 'absolute_detail_description',
            'relative_detail_description'
        ]

    def __call__(self, data):
        # ---- Select sample type (visual/text) for this batch ----
        selected_sample_type = self.use_prompt_sample_type

        if selected_sample_type == 'visual':
            data = [s['visual_prompt_sample'] for s in data]
        else:
            data = [s['text_prompt_sample'] for s in data]

        sample_type = data[0]['sample_type']

        images = [s['image'] for s in data]
        sizes = [s['size'] for s in data]
        # Must read the dataset-provided origin_size here: sample['size'] has
        # already been overwritten by PromptSegmentationResize with the resized
        # (letterbox) size, so copying it would make the decoder interpolate the
        # masks back to the resized size instead of the true original size.
        origin_sizes = [s['origin_size'] for s in data]

        pil_images = [s['pil_image'] for s in data]

        input_pil_images = []
        for per_pil_image in pil_images:
            w, h = per_pil_image.size
            factor = self.pil_resize / max(h, w)
            resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
            per_pil_image = per_pil_image.resize((resize_w, resize_h))
            input_pil_images.append(per_pil_image)

        input_images = []
        for i, per_image in enumerate(images):
            per_input_image = np.zeros((self.resize, self.resize, 3),
                                       dtype=np.float32)
            per_input_image[0:per_image.shape[0],
                            0:per_image.shape[1], :] = per_image
            per_input_image = torch.from_numpy(per_input_image)
            # [3,H,W]
            per_input_image = per_input_image.permute(2, 0, 1)
            input_images.append(per_input_image)
        input_images = torch.stack(input_images, dim=0)
        input_images = input_images.float()

        sizes = np.array(sizes, dtype=np.float32)
        origin_sizes = np.array(origin_sizes, dtype=np.float32)

        result = {
            'image': input_images,
            'size': sizes,
            'origin_size': origin_sizes,
            'sample_type': sample_type,
            'pil_image': input_pil_images,
        }

        if sample_type == 'visual':
            # ---- Batch-level visual prompt num (fixed) ----
            batch_target_num = self.use_visual_prompt_num
            for s in data:
                available = len(s['masks'])
                actual_num = min(batch_target_num, available)
                if actual_num < available:
                    indices = list(range(actual_num))
                    s['masks'] = [s['masks'][i] for i in indices]
                    s['prompt_points'] = [
                        s['prompt_points'][i] for i in indices
                    ]
                    s['prompt_boxes'] = [s['prompt_boxes'][i] for i in indices]
                    s['prompt_masks'] = [s['prompt_masks'][i] for i in indices]
                s['num_targets'] = len(s['masks'])

            # ---- Multi-target visual prompt ----
            all_masks = [s['masks'] for s in data]
            num_targets_list = [s['num_targets'] for s in data]

            # Pad GT masks: list of [N_i, H, W] tensors
            input_masks_list = []
            for per_sample_masks in all_masks:
                per_sample_input_masks = []
                for per_mask in per_sample_masks:
                    per_input_mask = np.zeros((self.resize, self.resize),
                                              dtype=np.float32)
                    per_input_mask[0:per_mask.shape[0],
                                   0:per_mask.shape[1]] = per_mask
                    per_input_mask = torch.from_numpy(per_input_mask)
                    per_sample_input_masks.append(per_input_mask)
                per_sample_input_masks = torch.stack(per_sample_input_masks,
                                                     dim=0)
                input_masks_list.append(per_sample_input_masks)

            result['mask'] = input_masks_list  # list of [N_i, H, W]
            result['num_targets'] = num_targets_list

            # ---- Select visual prompt type (fixed) ----
            selected_visual_prompt_type = self.use_visual_prompt_type

            # ---- Convert selected prompt type to unified mask ----
            input_prompt_masks_list = []

            if selected_visual_prompt_type == 'prompt_point':
                # Point → mask: mark point pixels, enhance with circles
                all_prompt_points = [s['prompt_points'] for s in data]
                point_kernel_size = 2 * self.point_radius + 1
                point_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (point_kernel_size, point_kernel_size))

                for sample_idx, per_sample_prompt_points in enumerate(
                        all_prompt_points):
                    per_sample_input_prompt_masks = []
                    for per_prompt_point in per_sample_prompt_points:
                        # Create mask at self.resize resolution
                        point_mask = np.zeros((self.resize, self.resize),
                                              dtype=np.uint8)
                        for pt_idx in range(per_prompt_point.shape[0]):
                            px = int(
                                np.clip(per_prompt_point[pt_idx, 0], 0,
                                        self.resize - 1))
                            py = int(
                                np.clip(per_prompt_point[pt_idx, 1], 0,
                                        self.resize - 1))
                            point_mask[py, px] = 1
                        # Enhance with circles (dilate)
                        point_mask = cv2.dilate(point_mask,
                                                point_kernel,
                                                iterations=1)
                        point_mask = point_mask.astype(np.float32)
                        per_sample_input_prompt_masks.append(
                            torch.from_numpy(point_mask))
                    per_sample_input_prompt_masks = torch.stack(
                        per_sample_input_prompt_masks, dim=0)
                    input_prompt_masks_list.append(
                        per_sample_input_prompt_masks)

            elif selected_visual_prompt_type == 'prompt_box':
                # Box → mask: fill box regions with 1
                all_prompt_boxes = [s['prompt_boxes'] for s in data]

                for sample_idx, per_sample_prompt_boxes in enumerate(
                        all_prompt_boxes):
                    per_sample_input_prompt_masks = []
                    for per_target_prompt_boxes in per_sample_prompt_boxes:
                        # Create mask at self.resize resolution
                        box_mask = np.zeros((self.resize, self.resize),
                                            dtype=np.float32)
                        for per_prompt_box in per_target_prompt_boxes:
                            x_min = int(
                                np.clip(per_prompt_box[0], 0, self.resize - 1))
                            y_min = int(
                                np.clip(per_prompt_box[1], 0, self.resize - 1))
                            x_max = int(
                                np.clip(per_prompt_box[2], 0, self.resize - 1))
                            y_max = int(
                                np.clip(per_prompt_box[3], 0, self.resize - 1))
                            box_mask[y_min:y_max + 1, x_min:x_max + 1] = 1
                        per_sample_input_prompt_masks.append(
                            torch.from_numpy(box_mask))
                    per_sample_input_prompt_masks = torch.stack(
                        per_sample_input_prompt_masks, dim=0)
                    input_prompt_masks_list.append(
                        per_sample_input_prompt_masks)

            elif selected_visual_prompt_type == 'prompt_mask':
                # Mask → mask: use existing noised prompt masks, pad to self.resize
                all_prompt_masks = [s['prompt_masks'] for s in data]

                for per_sample_prompt_masks in all_prompt_masks:
                    per_sample_input_prompt_masks = []
                    for per_prompt_mask in per_sample_prompt_masks:
                        padded = np.zeros((self.resize, self.resize),
                                          dtype=np.float32)
                        padded[0:per_prompt_mask.shape[0],
                               0:per_prompt_mask.shape[1]] = per_prompt_mask
                        per_sample_input_prompt_masks.append(
                            torch.from_numpy(padded))
                    per_sample_input_prompt_masks = torch.stack(
                        per_sample_input_prompt_masks, dim=0)
                    input_prompt_masks_list.append(
                        per_sample_input_prompt_masks)

            # list of [N_i, resize, resize]
            result['prompt_mask'] = input_prompt_masks_list
            result['visual_prompt_type'] = selected_visual_prompt_type
            result['visual_prompt_type_id'] = torch.full(
                (len(data), ),
                self.visual_prompt_type_to_id[selected_visual_prompt_type],
                dtype=torch.long)

            # ---- Raw text prompts for visual prompt (multi-region) ----
            # Each sample gets N_i <region> tokens joined by ", "
            prompt_texts = []
            for n in num_targets_list:
                prompt_texts.append(', '.join(['<region>'] * n))

            # ---- Select language type (fixed) ----
            prompt_languages = [self.use_prompt_language_type] * len(data)

            result['prompt_text'] = prompt_texts
            result['prompt_language'] = prompt_languages

        else:
            result['visual_prompt_type'] = None
            result['visual_prompt_type_id'] = None

            # ---- Batch-level text prompt num (fixed) ----

            batch_target_num = self.use_text_prompt_num
            for s in data:
                available = len(s['masks'])
                actual_num = min(batch_target_num, available)
                if actual_num < available:
                    indices = list(range(actual_num))
                    s['masks'] = [s['masks'][i] for i in indices]
                    s['prompt_texts'] = [s['prompt_texts'][i] for i in indices]
                s['num_targets'] = len(s['masks'])

            # ---- Multi-target text prompt ----
            all_masks = [s['masks'] for s in data]
            all_text_prompts = [s['prompt_texts'] for s in data]
            num_targets_list = [s['num_targets'] for s in data]

            # Pad GT masks: list of [N_i, H, W] tensors
            input_masks_list = []
            for per_sample_masks in all_masks:
                per_sample_input_masks = []
                for per_mask in per_sample_masks:
                    per_input_mask = np.zeros((self.resize, self.resize),
                                              dtype=np.float32)
                    per_input_mask[0:per_mask.shape[0],
                                   0:per_mask.shape[1]] = per_mask
                    per_input_mask = torch.from_numpy(per_input_mask)
                    per_sample_input_masks.append(per_input_mask)
                per_sample_input_masks = torch.stack(per_sample_input_masks,
                                                     dim=0)
                input_masks_list.append(per_sample_input_masks)

            # list of [N_i, H, W]
            result['mask'] = input_masks_list
            result['num_targets'] = num_targets_list

            # ---- Select language type and description type (fixed) ----
            batch_language = self.use_prompt_language_type
            selected_description_key = self.use_description_type

            # list of list of str
            origin_text_prompts = []
            prompt_languages = []
            for per_sample_text_prompts in all_text_prompts:
                per_sample_labels = []
                for per_mask_desc_dict in per_sample_text_prompts:
                    language_text_dict = per_mask_desc_dict[batch_language]
                    selected_text = language_text_dict[
                        selected_description_key]
                    label = selected_text.strip().lower()
                    per_sample_labels.append(label)
                origin_text_prompts.append(per_sample_labels)
                prompt_languages.append(batch_language)

            result['prompt_text'] = origin_text_prompts
            result['prompt_language'] = prompt_languages

        return result


def load_state_dict(saved_model_path, model, excluded_layer_name=()):
    '''
    saved_model_path: a saved model.state_dict() .pth file path
    model: a new defined model
    excluded_layer_name: layer names that doesn't want to load parameters
    '''
    if not saved_model_path:
        print('No pretrained model file!')
        return

    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'),
                                  weights_only=True)

    not_loaded_save_state_dict = []
    filtered_state_dict = {}
    for name, weight in saved_state_dict.items():
        if name in model.state_dict() and not any(
                excluded_name in name for excluded_name in excluded_layer_name
        ) and weight.shape == model.state_dict()[name].shape:
            filtered_state_dict[name] = weight
        else:
            not_loaded_save_state_dict.append(name)

    if len(filtered_state_dict) == 0:
        print('No pretrained parameters to load!')
    else:
        print(
            f'load/model weight nums:{len(filtered_state_dict)}/{len(model.state_dict())}'
        )
        print(f'not loaded save layer weight:\n{not_loaded_save_state_dict}')
        model.load_state_dict(filtered_state_dict, strict=False)

    return
