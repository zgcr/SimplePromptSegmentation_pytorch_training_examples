import os
import collections
import cv2
import json
import math
import numpy as np

from pycocotools import mask as mask_utils
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from torch.utils.data import Dataset


class SA1BDenseSegmentationDataset(Dataset):

    def __init__(self,
                 root_dir,
                 set_name=[
                     'sa_000000',
                 ],
                 set_type='train',
                 per_set_image_choose_max_num={
                     'sa_000000': 1000000,
                 },
                 per_image_mask_choose_max_num=200,
                 area_filter_ratio=0.0001,
                 transform=None):

        self.all_set_image_path_list = collections.OrderedDict()
        self.all_set_image_nums = collections.OrderedDict()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.process_set, root_dir, per_set_name, set_type):
                per_set_name
                for per_set_name in set_name
            }

            for future in tqdm(as_completed(futures), total=len(set_name)):
                per_set_name = futures[future]
                per_set_image_paths, per_set_count = future.result()
                self.all_set_image_nums[per_set_name] = per_set_count
                self.all_set_image_path_list[
                    per_set_name] = per_set_image_paths

        for key, value in self.all_set_image_path_list.items():
            print(f'set_name:{key},origin_image_num:{len(value)}')

        self.image_path_list = []
        for per_set_name, per_set_image_path_list in self.all_set_image_path_list.items(
        ):
            per_set_image_path_list = sorted(per_set_image_path_list)
            per_set_image_max_num = per_set_image_choose_max_num[per_set_name]
            if len(per_set_image_path_list) > per_set_image_max_num:
                per_set_image_path_list = per_set_image_path_list[
                    0:per_set_image_max_num]

            print(
                f'set_name:{per_set_name},choose_image_num:{len(per_set_image_path_list)}'
            )

            for per_image_info in per_set_image_path_list:
                self.image_path_list.append(per_image_info)
        self.image_path_list = sorted(self.image_path_list)

        self.per_image_mask_choose_max_num = per_image_mask_choose_max_num
        self.area_filter_ratio = area_filter_ratio
        self.transform = transform

        print(f'Dataset Size:{len(self.image_path_list)}')

    def process_set(self, root_dir, per_set_name, set_type):
        per_set_dir = os.path.join(root_dir, per_set_name, set_type)
        per_set_image_paths = []
        per_set_count = 0

        for root, folders, files in os.walk(per_set_dir):
            for file_name in files:
                if file_name.endswith('.jpg'):
                    per_image_path = os.path.join(root, file_name)

                    json_name = file_name.replace('.jpg', '.json')
                    per_json_path = os.path.join(root, json_name)

                    if os.path.exists(per_image_path) and os.path.exists(
                            per_json_path):
                        per_set_count += 1
                        per_set_image_paths.append([
                            file_name,
                            per_image_path,
                            per_json_path,
                        ])

        return per_set_image_paths, per_set_count

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        _, image_path, _ = self.image_path_list[idx]

        image = self.load_image(idx)
        image_boxes, image_masks = self.load_mask(idx)

        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)
        origin_size = size.copy()

        sample = {
            'path': image_path,
            'image': image,
            'box': image_boxes,
            'mask': image_masks,
            'scale': scale,
            'size': size,
            'origin_size': origin_size,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        _, image_path, _ = self.image_path_list[idx]

        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_mask(self, idx):
        _, _, json_path = self.image_path_list[idx]

        with open(json_path, encoding='utf-8') as f:
            json_data = json.load(f)

        image_h, image_w = json_data['image']['height'], json_data['image'][
            'width']
        annotations = json_data['annotations']

        if len(annotations) > self.per_image_mask_choose_max_num:
            annotations = annotations[:self.per_image_mask_choose_max_num]

        # SA-1B is class-agnostic, all masks are foreground class 0
        # box format: [x_min, y_min, x_max, y_max, class_id]
        target_boxes = np.zeros((0, 5))
        target_masks = np.zeros((image_h, image_w, 0))

        if len(annotations) == 0:
            return target_boxes.astype(np.float32), target_masks.astype(
                np.float32)

        # Use multi-threading to decode masks in parallel
        # (mask_utils.decode is C-based and releases the GIL)
        results = [None] * len(annotations)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.process_single_annot, annot, image_h, image_w):
                i
                for i, annot in enumerate(annotations)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        # Collect results in original order to ensure equivalence
        box_list = []
        mask_list = []
        for res in results:
            if res is not None:
                box_list.append(res[0])
                mask_list.append(res[1])

        if len(box_list) > 0:
            target_boxes = np.concatenate(box_list, axis=0)
            target_masks = np.stack(mask_list, axis=-1)

        assert target_boxes.shape[0] == target_masks.shape[-1]

        return target_boxes.astype(np.float32), target_masks.astype(np.float32)

    def process_single_annot(self, annot, image_h, image_w):
        """Process a single annotation, return (box, mask) or None."""
        # bbox format: [x_min, y_min, w, h]
        bbox = annot['bbox']

        x_min = math.ceil(max(bbox[0], 0))
        y_min = math.ceil(max(bbox[1], 0))
        x_max = math.ceil(min(bbox[0] + bbox[2], image_w))
        y_max = math.ceil(min(bbox[1] + bbox[3], image_h))
        box_w = x_max - x_min
        box_h = y_max - y_min

        if box_w <= 1 or box_h <= 1:
            return None

        # filter small masks by area ratio
        mask_area = annot['area']
        total_area = float(image_h * image_w)
        if mask_area / total_area < self.area_filter_ratio:
            return None

        # decode RLE mask
        rle_mask = annot['segmentation']
        decoded_mask = mask_utils.decode(rle_mask)
        decoded_mask[decoded_mask > 0] = 1

        # box: [x_min, y_min, x_max, y_max, class_id=0]
        target_box = np.array([[x_min, y_min, x_max, y_max, 0]])

        return target_box, decoded_mask


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

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

    from tools.path import interactive_segmentation_dataset_path

    import copy
    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleDenseSegmentation.dense_segmentation_common import DenseSegmentationResize, RandomHorizontalFlip, Normalize, DenseSegmentationTrainCollater

    sa1b_dataset = SA1BDenseSegmentationDataset(
        root_dir=interactive_segmentation_dataset_path,
        set_name=[
            'sa_000000',
        ],
        set_type='train',
        per_set_image_choose_max_num={
            'sa_000000': 1000000,
        },
        per_image_mask_choose_max_num=200,
        area_filter_ratio=0.0001,
        transform=transforms.Compose([
            DenseSegmentationResize(resize=1024,
                                    stride=32,
                                    multi_scale=False,
                                    multi_scale_range=[0.8, 1.0]),
            RandomHorizontalFlip(prob=0.5),
            # Normalize(mean=[123.675, 116.28, 103.53],
            #           std=[58.395, 57.12, 57.375]),
        ]))

    count = 0
    for per_sample in tqdm(sa1b_dataset):
        print('1111', per_sample['path'])
        print('1111', per_sample['image'].shape, per_sample['box'].shape,
              per_sample['mask'].shape, per_sample['scale'],
              per_sample['size'], per_sample['origin_size'])
        print('1111', per_sample['image'].dtype, per_sample['box'].dtype,
              per_sample['mask'].dtype, per_sample['scale'].dtype,
              per_sample['size'].dtype, per_sample['origin_size'].dtype)

        temp_dir = './temp1'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_not_draw = copy.deepcopy(image)
        mask = per_sample['mask']
        masks_num = mask.shape[2]

        # each instance gets a unique random color
        masks_instance_color = []
        for _ in range(masks_num):
            masks_instance_color.append(
                list(np.random.choice(range(256), size=3)))
        print("1212", masks_num, len(masks_instance_color),
              masks_instance_color[0] if masks_num > 0 else [])

        per_image_mask = np.zeros((image.shape[0], image.shape[1], 3))
        per_image_contours = []
        for i in range(masks_num):
            per_mask = mask[:, :, i]
            per_mask_color = np.array(
                (masks_instance_color[i][0], masks_instance_color[i][1],
                 masks_instance_color[i][2]))

            per_object_mask = np.nonzero(per_mask == 1.)
            per_image_mask[per_object_mask[0],
                           per_object_mask[1]] = per_mask_color

            # get contours
            new_per_image_mask = np.zeros((image.shape[0], image.shape[1]))
            new_per_image_mask[per_object_mask[0], per_object_mask[1]] = 255
            contours, _ = cv2.findContours(new_per_image_mask.astype(np.uint8),
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
            per_image_contours.append(contours)

        per_image_mask = per_image_mask.astype(np.uint8)
        per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_RGB2BGR)

        all_object_mask = np.nonzero(per_image_mask != 0)
        per_image_mask[all_object_mask[0],
                       all_object_mask[1]] = cv2.addWeighted(
                           image[all_object_mask[0], all_object_mask[1]], 0.5,
                           per_image_mask[all_object_mask[0],
                                          all_object_mask[1]], 1, 0)
        no_class_mask = np.nonzero(per_image_mask == 0)
        per_image_mask[no_class_mask[0],
                       no_class_mask[1]] = image[no_class_mask[0],
                                                 no_class_mask[1]]
        for contours in per_image_contours:
            cv2.drawContours(per_image_mask, contours, -1, (255, 255, 255), 2)

        cv2.imencode('.jpg', image_not_draw)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}.jpg'))
        cv2.imencode('.jpg', per_image_mask)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = DenseSegmentationTrainCollater(resize=1024, max_annots=200)
    train_loader = DataLoader(sa1b_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, masks, labels, sizes = data['image'], data['mask'], data[
            'label'], data['size']
        print('1111', images.shape, len(masks), len(labels), sizes.shape)

        for per_image_masks, per_image_labels in zip(masks, labels):
            print('2222', per_image_masks.shape, per_image_labels.shape)
            print('3333', per_image_labels)

        temp_dir = './temp2'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        images_np = images.permute(0, 2, 3, 1).cpu().numpy()

        for image_idx, (per_image, per_image_masks,
                        per_image_labels) in enumerate(
                            zip(images_np, masks, labels)):
            per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
            per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)
            per_image_not_draw = copy.deepcopy(per_image)
            per_image_masks = per_image_masks.permute(1, 2, 0).cpu().numpy()
            per_image_masks_num = per_image_masks.shape[2]

            # each instance gets a unique random color
            per_image_masks_instance_color = []
            for _ in range(per_image_masks_num):
                per_image_masks_instance_color.append(
                    list(np.random.choice(range(256), size=3)))
            print(
                "1212", per_image_masks_num,
                len(per_image_masks_instance_color),
                per_image_masks_instance_color[0]
                if per_image_masks_num > 0 else [])

            per_image_new_mask = np.zeros(
                (per_image.shape[0], per_image.shape[1], 3))
            per_image_contours = []
            for i in range(per_image_masks_num):
                per_mask = per_image_masks[:, :, i]
                per_mask_color = np.array(
                    (per_image_masks_instance_color[i][0],
                     per_image_masks_instance_color[i][1],
                     per_image_masks_instance_color[i][2]))

                per_object_mask = np.nonzero(per_mask == 1.)
                per_image_new_mask[per_object_mask[0],
                                   per_object_mask[1]] = per_mask_color

                # get contours
                new_per_image_mask = np.zeros(
                    (per_image.shape[0], per_image.shape[1]))
                new_per_image_mask[per_object_mask[0],
                                   per_object_mask[1]] = 255
                contours, _ = cv2.findContours(
                    new_per_image_mask.astype(np.uint8), cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)
                per_image_contours.append(contours)

            per_image_new_mask = per_image_new_mask.astype(np.uint8)
            per_image_new_mask = cv2.cvtColor(per_image_new_mask,
                                              cv2.COLOR_RGB2BGR)

            all_object_mask = np.nonzero(per_image_new_mask != 0)
            per_image_new_mask[all_object_mask[0],
                               all_object_mask[1]] = cv2.addWeighted(
                                   per_image[all_object_mask[0],
                                             all_object_mask[1]], 0.5,
                                   per_image_new_mask[all_object_mask[0],
                                                      all_object_mask[1]], 1,
                                   0)
            no_class_mask = np.nonzero(per_image_new_mask == 0)
            per_image_new_mask[no_class_mask[0],
                               no_class_mask[1]] = per_image[no_class_mask[0],
                                                             no_class_mask[1]]
            for contours in per_image_contours:
                cv2.drawContours(per_image_new_mask, contours, -1,
                                 (255, 255, 255), 2)

            cv2.imencode('.jpg', per_image_not_draw)[1].tofile(
                os.path.join(temp_dir, f'idx_{count}_{image_idx}.jpg'))
            cv2.imencode('.jpg', per_image_new_mask)[1].tofile(
                os.path.join(temp_dir, f'idx_{count}_{image_idx}_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break
