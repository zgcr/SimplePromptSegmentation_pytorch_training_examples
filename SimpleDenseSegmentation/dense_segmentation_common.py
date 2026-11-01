import cv2
import numpy as np

import torch
import torch.nn.functional as F


class DenseSegmentationResize:

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

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, boxes, masks, scale, size, origin_size = sample[
            'image'], sample['box'], sample['mask'], sample['scale'], sample[
                'size'], sample['origin_size']

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

        if masks.shape[2] > 0:
            masks = cv2.resize(masks, (resize_w, resize_h),
                               interpolation=cv2.INTER_NEAREST)
            if len(masks.shape) != 3:
                masks = np.expand_dims(masks, axis=-1)
        else:
            masks = np.zeros((resize_h, resize_w, 0), dtype=np.float32)

        factor = np.float32(factor)
        boxes[:, :4] *= factor
        scale *= factor
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample['image'], sample['box'], sample['mask'], sample[
            'scale'], sample['size'], sample[
                'origin_size'] = image, boxes, masks, scale, size, origin_size

        return sample


class RandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, boxes, masks, scale, size, origin_size = sample[
            'image'], sample['box'], sample['mask'], sample['scale'], sample[
                'size'], sample['origin_size']

        if boxes.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1, :]
            masks = masks[:, ::-1, :]

            _, w, _ = image.shape

            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()

            boxes[:, 0] = w - x2
            boxes[:, 2] = w - x1

        sample['image'], sample['box'], sample['mask'], sample[
            'scale'], sample['size'], sample[
                'origin_size'] = image, boxes, masks, scale, size, origin_size

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
        image, boxes, masks, scale, size, origin_size = sample[
            'image'], sample['box'], sample['mask'], sample['scale'], sample[
                'size'], sample['origin_size']

        image = (image - self.mean) / self.std

        sample['image'], sample['box'], sample['mask'], sample[
            'scale'], sample['size'], sample[
                'origin_size'] = image, boxes, masks, scale, size, origin_size

        return sample


class DenseSegmentationTrainCollater:

    def __init__(self, resize=1024, max_annots=200):
        self.resize = resize
        self.max_annots = max_annots

    def __call__(self, data):
        images = [s['image'] for s in data]
        boxes = [s['box'] for s in data]
        masks = [s['mask'] for s in data]
        sizes = [s['size'] for s in data]

        input_images = np.zeros((len(images), self.resize, self.resize, 3),
                                dtype=np.float32)
        for i, image in enumerate(images):
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image
        input_images = torch.from_numpy(input_images)
        # B H W 3 ->B 3 H W
        input_images = input_images.permute(0, 3, 1, 2)
        input_images = input_images.float()

        input_masks = []
        input_labels = []
        for i in range(len(images)):
            # (h_i, w_i, N_i)
            per_image_masks = masks[i].copy()
            # (N_i, 5)
            per_image_boxes = boxes[i].copy()

            assert per_image_masks.shape[-1] == per_image_boxes.shape[0]

            # 输出前景类别是从0开始,num_classes值才是背景,背景值loss计算时填充
            # (N_i, 5) -> (N_i,)
            per_mask_class_ids = per_image_boxes[:, -1].copy()
            # (N_i,)
            per_mask_class_ids = per_mask_class_ids.astype(np.float32)
            # (N_i,)
            per_mask_class_ids = torch.from_numpy(per_mask_class_ids).long()

            # (h_i, w_i, N_i) -> (N_i, h_i, w_i)
            per_image_masks = np.ascontiguousarray(
                per_image_masks.transpose(2, 0, 1))
            # (N_i, resize, resize)
            per_mask_input_class_masks = np.zeros(
                (len(per_image_masks), self.resize, self.resize),
                dtype=np.float32)
            # (N_i, h_i, w_i) -> pad into (N_i, resize, resize)
            per_mask_input_class_masks[:, 0:per_image_masks.shape[1],
                                       0:per_image_masks.
                                       shape[2]] = per_image_masks
            # (N_i, resize, resize)
            per_mask_input_class_masks = torch.from_numpy(
                per_mask_input_class_masks)
            # (N_i, resize, resize)
            per_mask_input_class_masks = per_mask_input_class_masks.float()

            assert len(per_mask_input_class_masks) == len(per_mask_class_ids)

            # 截断到 max_annots
            if len(per_mask_input_class_masks) > self.max_annots:
                per_mask_input_class_masks = per_mask_input_class_masks[:self.
                                                                        max_annots]
                per_mask_class_ids = per_mask_class_ids[:self.max_annots]

            assert len(per_mask_input_class_masks) == len(per_mask_class_ids)

            # append (N_i, resize, resize)
            input_masks.append(per_mask_input_class_masks)
            # append (N_i,)
            input_labels.append(per_mask_class_ids)

        sizes = np.array(sizes, dtype=np.float32)

        return {
            'image': input_images,
            'mask': input_masks,
            'label': input_labels,
            'size': sizes,
        }


class DenseSegmentationTestCollater:

    def __init__(self, resize=1024):
        self.resize = resize

    def __call__(self, data):
        images = [s['image'] for s in data]
        boxes = [s['box'] for s in data]
        masks = [s['mask'] for s in data]
        sizes = [s['size'] for s in data]
        origin_sizes = [s['origin_size'] for s in data]

        input_images = np.zeros((len(images), self.resize, self.resize, 3),
                                dtype=np.float32)
        for i, image in enumerate(images):
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image
        input_images = torch.from_numpy(input_images)
        # B H W 3 ->B 3 H W
        input_images = input_images.permute(0, 3, 1, 2)
        input_images = input_images.float()

        batch_boxes, batch_masks = [], []
        for i in range(len(images)):
            batch_boxes.append(torch.tensor(boxes[i], dtype=torch.float32))

            per_image_masks = np.zeros(
                (self.resize, self.resize, masks[i].shape[2]),
                dtype=np.float32)
            per_image_masks[0:masks[i].shape[0],
                            0:masks[i].shape[1], :] = masks[i]

            per_image_masks = per_image_masks.transpose(2, 0, 1)
            batch_masks.append(
                torch.tensor(per_image_masks, dtype=torch.float32))

        sizes = np.array(sizes, dtype=np.float32)
        origin_sizes = np.array(origin_sizes, dtype=np.float32)

        return {
            'image': input_images,
            'box': batch_boxes,
            'mask': batch_masks,
            'size': sizes,
            'origin_size': origin_sizes,
        }


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
