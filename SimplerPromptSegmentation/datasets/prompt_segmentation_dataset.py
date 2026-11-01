import os
import collections
import cv2
import math
import numpy as np
import orjson

from PIL import Image

from tqdm import tqdm
from pycocotools import mask as mask_utils
from concurrent.futures import ThreadPoolExecutor, as_completed

from torch.utils.data import Dataset


class PromptSegmentationDataset(Dataset):

    def __init__(
        self,
        #########################################################
        visual_prompt_image_root_dir='',
        visual_prompt_image_set_name=[
            'sa_000000',
        ],
        visual_prompt_image_set_type='train',
        visual_prompt_image_per_set_image_choose_max_num={
            'sa_000000': 1000000,
        },
        visual_prompt_per_image_mask_choose_max_num=32,
        visual_prompt_per_image_sample_num=1,
        visual_prompt_area_filter_ratio=0.001,
        visual_prompt_box_noise_wh_ratio=0.1,
        visual_prompt_mask_noise_area_ratio=0.04,
        #########################################################
        text_prompt_image_root_dir='',
        text_prompt_image_set_name=[
            'sa_000000',
        ],
        text_prompt_image_set_type='train',
        text_prompt_image_per_set_image_choose_max_num={
            'sa_000000': 1000000,
        },
        text_prompt_per_image_mask_choose_max_num=32,
        text_prompt_per_image_sample_num=1,
        text_prompt_area_filter_ratio=0.001,
        #########################################################
        transform=None):

        ########################################################################
        # load visual prompt image dataset
        ########################################################################
        self.all_visual_prompt_image_set_image_path_list = collections.OrderedDict(
        )
        self.all_visual_prompt_image_set_image_nums = collections.OrderedDict()

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(self.process_visual_prompt_image_set, visual_prompt_image_root_dir, per_set_name, visual_prompt_image_set_type):
                per_set_name
                for per_set_name in visual_prompt_image_set_name
            }

            for future in tqdm(as_completed(futures),
                               total=len(visual_prompt_image_set_name),
                               desc="Processing visual prompt image sets"):
                per_set_name = futures[future]
                per_set_image_paths, per_set_count = future.result()
                self.all_visual_prompt_image_set_image_nums[
                    per_set_name] = per_set_count
                self.all_visual_prompt_image_set_image_path_list[
                    per_set_name] = per_set_image_paths

        for key, value in self.all_visual_prompt_image_set_image_path_list.items(
        ):
            print(
                f'visual_prompt_image_set_name:{key},origin_image_num:{len(value)}'
            )

        # Image-level list: each element is [image_name, image_path, json_path]
        self.visual_prompt_image_path_list = []
        for per_set_name, per_set_image_path_list in self.all_visual_prompt_image_set_image_path_list.items(
        ):
            per_set_image_path_list = sorted(per_set_image_path_list)
            per_set_image_max_num = visual_prompt_image_per_set_image_choose_max_num[
                per_set_name]
            if len(per_set_image_path_list) > per_set_image_max_num:
                per_set_image_path_list = per_set_image_path_list[
                    0:per_set_image_max_num]

            print(
                f'visual_prompt_image_set_name:{per_set_name},choose_image_num:{len(per_set_image_path_list)}'
            )

            for per_image_info in per_set_image_path_list:
                self.visual_prompt_image_path_list.append(per_image_info)
        self.visual_prompt_image_path_list = sorted(
            self.visual_prompt_image_path_list)

        # get all valid image-mask pairs for visual prompt images
        self.visual_prompt_image_mask_pair_dict = collections.OrderedDict()

        batch_size = 10000
        results = [None] * len(self.visual_prompt_image_path_list)
        for batch_start in tqdm(
                range(0, len(self.visual_prompt_image_path_list), batch_size),
                desc="Processing visual prompt image mask pairs"):
            batch_end = min(batch_start + batch_size,
                            len(self.visual_prompt_image_path_list))
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {
                    executor.submit(
                        self.process_visual_prompt_image_mask_pairs, per_image_name, per_image_path, per_mask_label_path, visual_prompt_per_image_mask_choose_max_num, visual_prompt_area_filter_ratio):
                    idx
                    for idx,
                    (per_image_name, per_image_path,
                     per_mask_label_path) in enumerate(
                         self.
                         visual_prompt_image_path_list[batch_start:batch_end],
                         start=batch_start)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()

        total_visual_mask_pairs = 0
        for per_image_info, result in zip(self.visual_prompt_image_path_list,
                                          results):
            if result is not None and len(result) > 0:
                per_image_name = per_image_info[0]
                self.visual_prompt_image_mask_pair_dict[
                    per_image_name] = result
                total_visual_mask_pairs += len(result)

        # update image_path_list: keep only images with valid pairs
        filtered_visual_prompt_image_path_list = []
        for per_image_info in self.visual_prompt_image_path_list:
            per_image_name = per_image_info[0]
            if per_image_name in self.visual_prompt_image_mask_pair_dict and len(
                    self.visual_prompt_image_mask_pair_dict[per_image_name]
            ) >= 1:
                filtered_visual_prompt_image_path_list.append(per_image_info)
        self.visual_prompt_image_path_list = filtered_visual_prompt_image_path_list

        print(
            f'Visual prompt Image num:{len(self.visual_prompt_image_mask_pair_dict)}'
        )
        print(
            f'Visual prompt Image total mask pairs:{total_visual_mask_pairs}')

        ########################################################################
        # load text prompt image dataset
        ########################################################################
        self.all_text_prompt_image_set_image_path_list = collections.OrderedDict(
        )
        self.all_text_prompt_image_set_image_nums = collections.OrderedDict()

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(self.process_text_prompt_image_set, text_prompt_image_root_dir, per_set_name, text_prompt_image_set_type):
                per_set_name
                for per_set_name in text_prompt_image_set_name
            }

            for future in tqdm(as_completed(futures),
                               total=len(text_prompt_image_set_name),
                               desc="Processing text prompt image sets"):
                per_set_name = futures[future]
                per_set_image_paths, per_set_count = future.result()
                self.all_text_prompt_image_set_image_nums[
                    per_set_name] = per_set_count
                self.all_text_prompt_image_set_image_path_list[
                    per_set_name] = per_set_image_paths

        for key, value in self.all_text_prompt_image_set_image_path_list.items(
        ):
            print(
                f'text_prompt_image_set_name:{key},origin_image_num:{len(value)}'
            )

        # Image-level list: each element is [image_name, image_path, json_path]
        self.text_prompt_image_path_list = []
        for per_set_name, per_set_image_path_list in self.all_text_prompt_image_set_image_path_list.items(
        ):
            per_set_image_path_list = sorted(per_set_image_path_list)
            per_set_image_max_num = text_prompt_image_per_set_image_choose_max_num[
                per_set_name]
            if len(per_set_image_path_list) > per_set_image_max_num:
                per_set_image_path_list = per_set_image_path_list[
                    0:per_set_image_max_num]

            print(
                f'text_prompt_image_set_name:{per_set_name},choose_image_num:{len(per_set_image_path_list)}'
            )

            for per_image_info in per_set_image_path_list:
                self.text_prompt_image_path_list.append(per_image_info)
        self.text_prompt_image_path_list = sorted(
            self.text_prompt_image_path_list)

        # get all valid image-mask pairs for text prompt images
        self.text_prompt_image_mask_pair_dict = collections.OrderedDict()

        batch_size = 10000
        results = [None] * len(self.text_prompt_image_path_list)
        for batch_start in tqdm(
                range(0, len(self.text_prompt_image_path_list), batch_size),
                desc="Processing text prompt image mask pairs"):
            batch_end = min(batch_start + batch_size,
                            len(self.text_prompt_image_path_list))
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {
                    executor.submit(self.process_text_prompt_image_mask_pairs, per_image_name, per_image_path, per_mask_label_path, text_prompt_per_image_mask_choose_max_num, text_prompt_area_filter_ratio):
                    idx
                    for idx,
                    (per_image_name, per_image_path,
                     per_mask_label_path) in enumerate(
                         self.
                         text_prompt_image_path_list[batch_start:batch_end],
                         start=batch_start)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()

        total_text_mask_pairs = 0
        for per_image_info, result in zip(self.text_prompt_image_path_list,
                                          results):
            if result is not None and len(result) > 0:
                per_image_name = per_image_info[0]
                self.text_prompt_image_mask_pair_dict[per_image_name] = result
                total_text_mask_pairs += len(result)

        # update image_path_list: keep only images with valid pairs
        filtered_text_prompt_image_path_list = []
        for per_image_info in self.text_prompt_image_path_list:
            per_image_name = per_image_info[0]
            if per_image_name in self.text_prompt_image_mask_pair_dict and len(
                    self.text_prompt_image_mask_pair_dict[per_image_name]
            ) >= 1:
                filtered_text_prompt_image_path_list.append(per_image_info)
        self.text_prompt_image_path_list = filtered_text_prompt_image_path_list

        print(
            f'Text prompt Image num:{len(self.text_prompt_image_mask_pair_dict)}'
        )
        print(f'Text prompt Image total mask pairs:{total_text_mask_pairs}')

        ########################################################################
        # common params
        ########################################################################
        self.visual_prompt_per_image_sample_num = visual_prompt_per_image_sample_num
        self.text_prompt_per_image_sample_num = text_prompt_per_image_sample_num

        self.visual_prompt_area_filter_ratio = visual_prompt_area_filter_ratio
        self.visual_prompt_box_noise_wh_ratio = visual_prompt_box_noise_wh_ratio
        self.visual_prompt_mask_noise_area_ratio = visual_prompt_mask_noise_area_ratio

        self.transform = transform

        print(
            f'Visual prompt Dataset Size:{len(self.visual_prompt_image_path_list)}'
        )
        print(
            f'Text prompt Dataset Size:{len(self.text_prompt_image_path_list)}'
        )

    def process_visual_prompt_image_set(self, root_dir, per_set_name,
                                        set_type):
        per_set_dir = os.path.join(root_dir, per_set_name, set_type)
        per_set_image_paths = []
        per_set_count = 0

        with os.scandir(per_set_dir) as per_set_dir_entries:
            per_set_file_names = set(per_entry.name
                                     for per_entry in per_set_dir_entries)

        for file_name in per_set_file_names:
            if file_name.endswith('.jpg'):
                per_image_path = os.path.join(per_set_dir, file_name)

                json_name = file_name.replace('.jpg', '.json')
                per_mask_label_path = os.path.join(per_set_dir, json_name)

                if json_name in per_set_file_names:
                    per_set_count += 1
                    per_set_image_paths.append([
                        file_name,
                        per_image_path,
                        per_mask_label_path,
                    ])

        return per_set_image_paths, per_set_count

    def process_visual_prompt_image_mask_pairs(self, per_image_name,
                                               per_image_path,
                                               per_mask_label_path,
                                               per_image_mask_choose_max_num,
                                               area_filter_ratio):
        result_list = []
        with open(per_mask_label_path, 'rb') as f:
            per_image_json_data = orjson.loads(f.read())

        per_image_annotation = per_image_json_data['annotations']

        per_image_h, per_image_w = per_image_json_data['image'][
            'height'], per_image_json_data['image']['width']

        for mask_list_idx, per_annot in enumerate(per_image_annotation):
            # bbox format:[x_min, y_min, w, h]
            per_box = per_annot['bbox']

            x_min = math.ceil(max(per_box[0], 0))
            y_min = math.ceil(max(per_box[1], 0))
            x_max = math.ceil(min(per_box[0] + per_box[2], per_image_w))
            y_max = math.ceil(min(per_box[1] + per_box[3], per_image_h))
            box_w = math.ceil(x_max - x_min)
            box_h = math.ceil(y_max - y_min)

            if box_w / per_image_w < math.sqrt(
                    area_filter_ratio) and box_h / per_image_h < math.sqrt(
                        area_filter_ratio):
                continue

            if (box_w * box_h) / float(
                    per_image_h * per_image_w) < area_filter_ratio:
                continue

            if per_annot['area'] / float(
                    per_image_h * per_image_w
            ) < area_filter_ratio or per_annot['area'] / float(
                    per_image_h * per_image_w) > 0.9:
                continue

            result_list.append([
                per_image_name,
                mask_list_idx,
                per_image_path,
                per_mask_label_path,
                per_image_h,
                per_image_w,
            ])

        if len(result_list) > per_image_mask_choose_max_num:
            result_list = result_list[0:per_image_mask_choose_max_num]

        return result_list

    def process_text_prompt_image_set(self, root_dir, per_set_name, set_type):
        per_set_dir = os.path.join(root_dir, per_set_name, set_type)
        per_set_image_paths = []
        per_set_count = 0

        per_sub_dir_path_list = [per_set_dir]
        while len(per_sub_dir_path_list) > 0:
            per_sub_dir_path = per_sub_dir_path_list.pop()

            per_sub_dir_file_names = set()
            with os.scandir(per_sub_dir_path) as per_sub_dir_entries:
                for per_entry in per_sub_dir_entries:
                    if per_entry.is_dir():
                        per_sub_dir_path_list.append(per_entry.path)
                    else:
                        per_sub_dir_file_names.add(per_entry.name)

            for file_name in per_sub_dir_file_names:
                if file_name.endswith('.jpg'):
                    per_image_path = os.path.join(per_sub_dir_path, file_name)

                    json_name = file_name.replace('.jpg', '.json')
                    per_mask_label_path = os.path.join(per_sub_dir_path,
                                                       json_name)

                    if json_name in per_sub_dir_file_names:
                        per_set_count += 1
                        per_set_image_paths.append([
                            file_name,
                            per_image_path,
                            per_mask_label_path,
                        ])

        return per_set_image_paths, per_set_count

    def process_text_prompt_image_mask_pairs(self, per_image_name,
                                             per_image_path,
                                             per_mask_label_path,
                                             per_image_mask_choose_max_num,
                                             area_filter_ratio):
        result_list = []
        per_image_dir = os.path.dirname(per_image_path)

        with open(per_mask_label_path, 'rb') as f:
            per_image_json_data = orjson.loads(f.read())

        with os.scandir(per_image_dir) as per_image_dir_entries:
            per_image_dir_file_names = set(
                per_entry.name for per_entry in per_image_dir_entries)

        per_image_mask_items = list(per_image_json_data.items())

        for per_mask_name, per_mask_desc in per_image_mask_items:
            per_mask_path = os.path.join(per_image_dir, per_mask_name)
            if per_mask_name not in per_image_dir_file_names:
                continue

            # read mask_box, mask_area, mask_h, mask_w from json instead of
            # loading mask image or reading image header, mask_h/mask_w are
            # already checked equal to image_h/image_w when generating dataset
            # [x_min, y_min, w, h]
            per_box = per_mask_desc['mask_box']
            per_mask_area = per_mask_desc['mask_area']
            per_image_h = per_mask_desc['mask_h']
            per_image_w = per_mask_desc['mask_w']

            x_min = math.ceil(max(per_box[0], 0))
            y_min = math.ceil(max(per_box[1], 0))
            x_max = math.ceil(min(per_box[0] + per_box[2], per_image_w))
            y_max = math.ceil(min(per_box[1] + per_box[3], per_image_h))
            box_w = math.ceil(x_max - x_min)
            box_h = math.ceil(y_max - y_min)

            if box_w / per_image_w < math.sqrt(
                    area_filter_ratio) and box_h / per_image_h < math.sqrt(
                        area_filter_ratio):
                continue

            if (box_w * box_h) / float(
                    per_image_h * per_image_w) < area_filter_ratio:
                continue

            if per_mask_area / float(
                    per_image_h *
                    per_image_w) < area_filter_ratio or per_mask_area / float(
                        per_image_h * per_image_w) > 0.9:
                continue

            result_list.append([
                per_image_name,
                per_mask_name,
                per_image_path,
                per_mask_path,
                per_mask_label_path,
                per_image_h,
                per_image_w,
            ])

        if len(result_list) > per_image_mask_choose_max_num:
            result_list = result_list[0:per_image_mask_choose_max_num]

        return result_list

    def __len__(self):
        return len(self.text_prompt_image_path_list)

    def __getitem__(self, idx):
        ########################################################################
        # get visual prompt image data
        ########################################################################
        visual_image_idx = np.random.choice(
            len(self.visual_prompt_image_path_list))
        visual_image_name, visual_image_path, visual_json_path = self.visual_prompt_image_path_list[
            visual_image_idx]
        visual_image_mask_pairs = self.visual_prompt_image_mask_pair_dict[
            visual_image_name]

        # Sample random N pairs from valid image-mask pairs (1 to max)
        visual_sample_num = min(self.visual_prompt_per_image_sample_num,
                                len(visual_image_mask_pairs))
        sampled_visual_image_mask_pair_indices = np.random.choice(
            len(visual_image_mask_pairs),
            size=visual_sample_num,
            replace=False).tolist()

        # Load image
        visual_image = self.load_visual_prompt_image(visual_image_idx)
        visual_pil_image = self.load_pil_visual_prompt_image(visual_image_idx)

        # Load multiple masks from JSON
        with open(visual_json_path, 'rb') as f:
            per_image_json_data = orjson.loads(f.read())
        visual_image_annotations = per_image_json_data['annotations']

        visual_masks = []
        visual_prompt_points = []
        visual_prompt_boxes = []
        visual_prompt_masks = []
        for pair_idx in sampled_visual_image_mask_pair_indices:
            target_mask = self.load_visual_prompt_mask(
                visual_image_annotations, visual_image_mask_pairs, pair_idx)
            visual_masks.append(target_mask)

            # Generate prompt point from GT mask
            prompt_point = self.load_points((target_mask
                                             > 0.5).astype(np.float32))
            visual_prompt_points.append(prompt_point)

            # Generate prompt boxes from GT mask
            prompt_boxes = self.load_box((target_mask
                                          > 0.5).astype(np.float32))
            visual_prompt_boxes.append(prompt_boxes)

            # Generate noised prompt mask from GT mask
            prompt_mask = self.noise_mask((target_mask
                                           > 0.2).astype(np.float32))
            visual_prompt_masks.append(prompt_mask)

        visual_size = np.array([visual_image.shape[0],
                                visual_image.shape[1]]).astype(np.float32)
        origin_visual_size = np.array(
            [visual_image.shape[0], visual_image.shape[1]]).astype(np.float32)

        visual_prompt_sample = {
            'image_path': visual_image_path,
            'image': visual_image,
            'pil_image': visual_pil_image,
            'masks': visual_masks,
            'size': visual_size,
            'origin_size': origin_visual_size,
            'prompt_points': visual_prompt_points,
            'prompt_boxes': visual_prompt_boxes,
            'prompt_masks': visual_prompt_masks,
            'num_targets': visual_sample_num,
            'sample_type': 'visual',
        }

        if self.transform:
            visual_prompt_sample = self.transform(visual_prompt_sample)

        ########################################################################
        # get text prompt image data
        ########################################################################
        text_image_idx = idx
        text_image_name, text_image_path, text_json_path = self.text_prompt_image_path_list[
            text_image_idx]
        text_image_mask_pairs = self.text_prompt_image_mask_pair_dict[
            text_image_name]

        # Sample random N pairs from valid image-mask pairs (1 to max)
        text_sample_num = min(self.text_prompt_per_image_sample_num,
                              len(text_image_mask_pairs))
        sampled_text_image_mask_pair_indices = np.random.choice(
            len(text_image_mask_pairs), size=text_sample_num,
            replace=False).tolist()

        # Load image
        text_image = self.load_text_prompt_image(text_image_idx)
        text_pil_image = self.load_pil_text_prompt_image(text_image_idx)

        with open(text_json_path, 'rb') as f:
            per_text_json_data = orjson.loads(f.read())

        # Load multiple masks and text descriptions
        text_masks = []
        text_prompt_texts = []
        for pair_idx in sampled_text_image_mask_pair_indices:
            target_mask = self.load_text_prompt_mask(text_image_mask_pairs,
                                                     pair_idx)
            text_masks.append(target_mask)

            _, per_mask_name, _, _, _, _, _ = text_image_mask_pairs[pair_idx]
            text_description = per_text_json_data[per_mask_name]
            text_prompt_texts.append(text_description)

        text_size = np.array([text_image.shape[0],
                              text_image.shape[1]]).astype(np.float32)
        origin_text_size = np.array([text_image.shape[0],
                                     text_image.shape[1]]).astype(np.float32)

        text_prompt_sample = {
            'image_path': text_image_path,
            'image': text_image,
            'pil_image': text_pil_image,
            'masks': text_masks,
            'size': text_size,
            'origin_size': origin_text_size,
            'prompt_texts': text_prompt_texts,
            'num_targets': text_sample_num,
            'sample_type': 'text',
        }

        if self.transform:
            text_prompt_sample = self.transform(text_prompt_sample)

        return {
            'visual_prompt_sample': visual_prompt_sample,
            'text_prompt_sample': text_prompt_sample,
        }

    def load_visual_prompt_image(self, idx):
        _, per_image_path, _ = self.visual_prompt_image_path_list[idx]

        image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_pil_visual_prompt_image(self, idx):
        _, per_image_path, _ = self.visual_prompt_image_path_list[idx]

        pil_image = Image.open(per_image_path).convert('RGB')

        return pil_image

    def load_visual_prompt_mask(self, per_image_annotations,
                                per_image_mask_pairs, pair_idx):
        _, mask_list_idx, _, _, _, _ = per_image_mask_pairs[pair_idx]

        per_annot = per_image_annotations[mask_list_idx]
        target_mask = mask_utils.decode(per_annot['segmentation'])
        target_mask[target_mask > 0] = 1

        return target_mask.astype(np.float32)

    def load_points(self, mask):
        point_label = 1
        mask_h, mask_w = mask.shape[0], mask.shape[1]
        area_threshold = self.visual_prompt_area_filter_ratio * mask_h * mask_w

        # erode mask by 10 pixels to avoid sampling points near edges
        erode_radius = 10
        erode_kernel_size = 2 * erode_radius + 1
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
        eroded_mask = cv2.erode(mask.astype(np.uint8),
                                erode_kernel,
                                iterations=1).astype(np.float32)

        # connected component analysis on eroded mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            eroded_mask.astype(np.uint8), connectivity=8)

        # collect large-area connected components (skip background label 0)
        large_components = []
        for label_id in range(1, num_labels):
            component_area = stats[label_id, cv2.CC_STAT_AREA]
            if component_area > area_threshold:
                large_components.append(label_id)

        points = []
        max_retries = 100
        if len(large_components) > 0:
            # sample one point per large connected component on eroded mask
            for label_id in large_components:
                component_coords = np.argwhere(labels == label_id)

                centroid_y = np.mean(component_coords[:, 0])
                centroid_x = np.mean(component_coords[:, 1])

                min_y = np.min(component_coords[:, 0])
                max_y = np.max(component_coords[:, 0])
                min_x = np.min(component_coords[:, 1])
                max_x = np.max(component_coords[:, 1])

                bbox_h = max_y - min_y
                bbox_w = max_x - min_x
                offset_radius = min(bbox_h, bbox_w) * 0.5

                sampled = False
                for _ in range(max_retries):
                    angle = np.random.uniform(0, 2 * math.pi)
                    dist = np.random.uniform(0, offset_radius)
                    offset_y = dist * math.cos(angle)
                    offset_x = dist * math.sin(angle)

                    point_y = int(np.clip(centroid_y + offset_y, min_y, max_y))
                    point_x = int(np.clip(centroid_x + offset_x, min_x, max_x))

                    if eroded_mask[point_y, point_x] > 0:
                        points.append([point_x, point_y, point_label])
                        sampled = True
                        break

                if not sampled:
                    # fallback: randomly pick from component foreground pixels
                    fallback_idx = np.random.choice(len(component_coords))
                    points.append([
                        component_coords[fallback_idx][1],
                        component_coords[fallback_idx][0],
                        point_label,
                    ])
        else:
            # all components are small, sample one point on the original mask
            all_point_coords = np.argwhere(mask != 0)

            centroid_y = np.mean(all_point_coords[:, 0])
            centroid_x = np.mean(all_point_coords[:, 1])

            min_y = np.min(all_point_coords[:, 0])
            max_y = np.max(all_point_coords[:, 0])
            min_x = np.min(all_point_coords[:, 1])
            max_x = np.max(all_point_coords[:, 1])

            bbox_h = max_y - min_y
            bbox_w = max_x - min_x
            offset_radius = min(bbox_h, bbox_w) * 0.5

            sampled = False
            for _ in range(max_retries):
                angle = np.random.uniform(0, 2 * math.pi)
                dist = np.random.uniform(0, offset_radius)
                offset_y = dist * math.cos(angle)
                offset_x = dist * math.sin(angle)

                point_y = int(np.clip(centroid_y + offset_y, min_y, max_y))
                point_x = int(np.clip(centroid_x + offset_x, min_x, max_x))

                if mask[point_y, point_x] > 0:
                    points.append([point_x, point_y, point_label])
                    sampled = True
                    break

            if not sampled:
                # fallback: randomly pick from mask foreground pixels
                fallback_idx = np.random.choice(len(all_point_coords))
                points.append([
                    all_point_coords[fallback_idx][1],
                    all_point_coords[fallback_idx][0],
                    point_label,
                ])

        points = np.array(points, dtype=np.float32)

        return points

    def load_box(self, mask):
        h, w = mask.shape[0], mask.shape[1]
        area_threshold = self.visual_prompt_area_filter_ratio * h * w

        mask_uint8 = (mask > 0).astype(np.uint8)

        # connected component analysis on mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8)

        # collect large-area connected components (skip background label 0)
        large_components = []
        for label_id in range(1, num_labels):
            component_area = stats[label_id, cv2.CC_STAT_AREA]
            if component_area > area_threshold:
                large_components.append(label_id)

        boxes = []
        mask_size = [h, w]

        if len(large_components) > 0:
            # get bbox for each large connected component and add noise
            for label_id in large_components:
                component_x = stats[label_id, cv2.CC_STAT_LEFT]
                component_y = stats[label_id, cv2.CC_STAT_TOP]
                component_w = stats[label_id, cv2.CC_STAT_WIDTH]
                component_h = stats[label_id, cv2.CC_STAT_HEIGHT]

                x_min = component_x
                y_min = component_y
                x_max = component_x + component_w - 1
                y_max = component_y + component_h - 1

                box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                box = self.noise_box(box, mask_size)
                boxes.append(box)
        else:
            # all components are small, get bbox for the whole mask and add noise
            mask_bool = mask.astype(bool)

            if mask_bool.any():
                xs = np.arange(w, dtype=np.int32)
                ys = np.arange(h, dtype=np.int32)
                grid_xs, grid_ys = np.meshgrid(xs, ys, indexing='xy')

                x_min = np.min(np.where(mask_bool, grid_xs, w))
                y_min = np.min(np.where(mask_bool, grid_ys, h))
                x_max = np.max(np.where(mask_bool, grid_xs, -1))
                y_max = np.max(np.where(mask_bool, grid_ys, -1))
            else:
                x_min, y_min, x_max, y_max = w, h, -1, -1

            box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            box = self.noise_box(box, mask_size)
            boxes.append(box)

        boxes = np.array(boxes, dtype=np.float32)

        return boxes

    def noise_box(self, properties_box, mask_np_shape):
        if -1 in properties_box:
            return properties_box.astype(np.float32)

        w, h = properties_box[2] - properties_box[0], properties_box[
            3] - properties_box[1]

        if h / mask_np_shape[0] <= math.sqrt(
                self.visual_prompt_area_filter_ratio
        ) or w / mask_np_shape[1] <= math.sqrt(
                self.visual_prompt_area_filter_ratio):
            return properties_box.astype(np.float32)

        noise_x, noise_y = int(w * self.visual_prompt_box_noise_wh_ratio), int(
            h * self.visual_prompt_box_noise_wh_ratio)
        noise_x, noise_y = min(int(mask_np_shape[1] * 0.02),
                               noise_x), min(int(mask_np_shape[0] * 0.02),
                                             noise_y)

        if noise_x <= 1 or noise_y <= 1:
            return properties_box.astype(np.float32)

        x0 = properties_box[0] + max(
            min(np.random.randint(-noise_x, noise_x), w / 2), -w / 2)
        y0 = properties_box[1] + max(
            min(np.random.randint(-noise_y, noise_y), h / 2), -h / 2)
        x1 = properties_box[2] + max(
            min(np.random.randint(-noise_x, noise_x), w / 2), -w / 2)
        y1 = properties_box[3] + max(
            min(np.random.randint(-noise_y, noise_y), h / 2), -h / 2)

        x0 = x0 if x0 >= 0 else 0
        y0 = y0 if y0 >= 0 else 0
        x1 = x1 if x1 <= mask_np_shape[1] else mask_np_shape[1]
        y1 = y1 if y1 <= mask_np_shape[0] else mask_np_shape[0]

        post_properties_box = np.array([x0, y0, x1, y1])
        post_properties_box = np.where(post_properties_box > 0,
                                       post_properties_box, 0)

        if x0 >= x1 or y0 >= y1:
            return properties_box.astype(np.float32)
        else:
            return post_properties_box.astype(np.float32)

    def noise_mask(self, properties_mask):
        mask_h, mask_w = properties_mask.shape[0], properties_mask.shape[1]

        origin_mask_area = np.count_nonzero(properties_mask)
        total_mask_area = float(mask_h * mask_w)

        mask_area_ratio = origin_mask_area / total_mask_area

        if mask_area_ratio < self.visual_prompt_area_filter_ratio:
            return properties_mask.astype(np.float32)

        reduce_mask_area = origin_mask_area * self.visual_prompt_mask_noise_area_ratio
        reduce_area_ratio = reduce_mask_area / total_mask_area

        if reduce_area_ratio < self.visual_prompt_area_filter_ratio:
            return properties_mask.astype(np.float32)

        max_kernel = np.sqrt(reduce_mask_area) / 2.
        if int(max_kernel) > 1:
            kernel = np.random.randint(1, max_kernel)
            kernel = np.ones((kernel, kernel), np.uint8)
            if np.random.uniform(0, 1) < 0.5:
                post_properties_mask = cv2.erode(properties_mask,
                                                 kernel,
                                                 iterations=1)
            else:
                post_properties_mask = cv2.dilate(properties_mask,
                                                  kernel,
                                                  iterations=1)
        else:
            post_properties_mask = properties_mask

        if np.count_nonzero(
                post_properties_mask
        ) / total_mask_area > self.visual_prompt_area_filter_ratio:
            return post_properties_mask.astype(np.float32)
        else:
            return properties_mask.astype(np.float32)

    def load_text_prompt_image(self, idx):
        _, per_image_path, _ = self.text_prompt_image_path_list[idx]

        image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_pil_text_prompt_image(self, idx):
        _, per_image_path, _ = self.text_prompt_image_path_list[idx]

        pil_image = Image.open(per_image_path).convert('RGB')

        return pil_image

    def load_text_prompt_mask(self, per_image_mask_pairs, pair_idx):
        _, _, _, per_mask_path, _, _, _ = per_image_mask_pairs[pair_idx]

        target_mask = np.array(Image.open(per_mask_path).convert('L'),
                               dtype=np.uint8)
        target_mask[target_mask > 0] = 1

        return target_mask.astype(np.float32)


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

    from tools.path import interactive_segmentation_dataset_path, text_prompt_segmentation_dataset_path

    import copy
    import torchvision.transforms as transforms
    from tqdm import tqdm

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
        visual_prompt_per_image_sample_num=1,
        visual_prompt_area_filter_ratio=0.001,
        visual_prompt_box_noise_wh_ratio=0.1,
        visual_prompt_mask_noise_area_ratio=0.04,
        #########################################################
        text_prompt_image_root_dir=text_prompt_segmentation_dataset_path,
        text_prompt_image_set_name=[
            'sa_000000',
        ],
        text_prompt_image_set_type='train',
        text_prompt_image_per_set_image_choose_max_num={
            'sa_000000': 1000000,
        },
        text_prompt_per_image_mask_choose_max_num=32,
        text_prompt_per_image_sample_num=1,
        text_prompt_area_filter_ratio=0.001,
        #########################################################
        transform=transforms.Compose([
            PromptSegmentationResize(resize=1024,
                                     stride=32,
                                     multi_scale=False,
                                     multi_scale_range=[0.8, 1.0]),
            # Normalize(mean=[123.675, 116.28, 103.53],
            #           std=[58.395, 57.12, 57.375]),
        ]))

    count = 0
    for per_sample in tqdm(prompt_segmentation_dataset):
        visual_sample = per_sample['visual_prompt_sample']
        text_sample = per_sample['text_prompt_sample']

        ################################################################
        # print visual prompt sample info
        ################################################################
        print('1111', visual_sample['image_path'])
        print('1212', visual_sample['image'].shape, visual_sample['size'],
              visual_sample['num_targets'], visual_sample['sample_type'])
        print('1313', visual_sample['image'].dtype,
              visual_sample['size'].dtype)
        print('1414', len(visual_sample['masks']),
              len(visual_sample['prompt_points']),
              len(visual_sample['prompt_boxes']),
              len(visual_sample['prompt_masks']))
        print('1515', visual_sample['pil_image'].size,
              visual_sample['pil_image'].mode)

        for per_mask in visual_sample['masks']:
            print('2222', per_mask.shape, per_mask.dtype)
        for per_prompt_point in visual_sample['prompt_points']:
            print('3333', per_prompt_point.shape, per_prompt_point)
        for per_target_prompt_boxes in visual_sample['prompt_boxes']:
            print('4444', per_target_prompt_boxes.shape,
                  per_target_prompt_boxes)
        for per_prompt_mask in visual_sample['prompt_masks']:
            print('5555', per_prompt_mask.shape, per_prompt_mask.dtype)

        ################################################################
        # visualize visual prompt sample
        ################################################################
        temp_dir = f'./temp1'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        visual_image = np.ascontiguousarray(visual_sample['image'],
                                            dtype=np.uint8)
        visual_image = cv2.cvtColor(visual_image, cv2.COLOR_RGB2BGR)
        # save visual prompt pil_image (original resolution)
        visual_pil_image = visual_sample['pil_image']

        visual_masks = visual_sample['masks']
        visual_masks_num = len(visual_masks)

        # draw all masks on one image
        image_for_visual_mask = copy.deepcopy(visual_image).astype(np.uint8)
        per_image_visual_mask = np.zeros((image_for_visual_mask.shape[0],
                                          image_for_visual_mask.shape[1], 3))
        per_image_visual_contours = []
        for i in range(visual_masks_num):
            per_mask = visual_masks[i]
            mask_color = [int(np.random.choice(range(256))) for _ in range(3)]
            per_object_mask = np.nonzero(per_mask == 1.)
            per_image_visual_mask[per_object_mask[0],
                                  per_object_mask[1]] = mask_color

            new_per_image_visual_mask = np.zeros(
                (image_for_visual_mask.shape[0],
                 image_for_visual_mask.shape[1]))
            new_per_image_visual_mask[per_object_mask[0],
                                      per_object_mask[1]] = 255
            contours, _ = cv2.findContours(
                new_per_image_visual_mask.astype(np.uint8), cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            per_image_visual_contours.append(contours)

        per_image_visual_mask = per_image_visual_mask.astype(np.uint8)
        per_image_visual_mask = cv2.cvtColor(per_image_visual_mask,
                                             cv2.COLOR_RGB2BGR)
        all_classes_mask = np.nonzero(per_image_visual_mask != 0)
        if len(all_classes_mask[0]) > 0:
            per_image_visual_mask[
                all_classes_mask[0], all_classes_mask[1]] = cv2.addWeighted(
                    image_for_visual_mask[all_classes_mask[0],
                                          all_classes_mask[1]], 0.5,
                    per_image_visual_mask[all_classes_mask[0],
                                          all_classes_mask[1]], 1, 0)
        no_class_mask = np.nonzero(per_image_visual_mask == 0)
        if len(no_class_mask[0]) > 0:
            per_image_visual_mask[no_class_mask[0],
                                  no_class_mask[1]] = image_for_visual_mask[
                                      no_class_mask[0], no_class_mask[1]]
        for contours in per_image_visual_contours:
            cv2.drawContours(per_image_visual_mask, contours, -1,
                             (255, 255, 255), 2)

        cv2.imencode('.jpg', visual_image)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}_visual_image.jpg'))
        cv2.imencode('.jpg', per_image_visual_mask)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}_visual_image_with_mask.jpg'))
        visual_pil_image.save(
            os.path.join(temp_dir, f'idx_{count}_visual_pil_image.jpg'))

        # visualize visual prompt point, prompt box, prompt mask
        prompt_points = visual_sample['prompt_points']
        prompt_boxes = visual_sample['prompt_boxes']
        prompt_masks = visual_sample['prompt_masks']
        positive_prompt_point_color = [
            int(np.random.choice(range(256))) for _ in range(3)
        ]
        negative_prompt_point_color = [
            int(np.random.choice(range(256))) for _ in range(3)
        ]

        image_for_visual_prompt_box = copy.deepcopy(visual_image)

        for per_prompt_point in prompt_points:
            for per_point in per_prompt_point:
                point_label = per_point[2]
                if point_label == 1:
                    cv2.circle(image_for_visual_prompt_box,
                               (int(per_point[0]), int(per_point[1])), 10,
                               positive_prompt_point_color, -1)
                elif point_label == 0:
                    cv2.circle(image_for_visual_prompt_box,
                               (int(per_point[0]), int(per_point[1])), 10,
                               negative_prompt_point_color, -1)

        box_global_idx = 0
        for per_target_prompt_boxes in prompt_boxes:
            for per_prompt_box in per_target_prompt_boxes:
                prompt_box_color = [
                    int(np.random.choice(range(256))) for _ in range(3)
                ]
                per_image_prompt_box = (per_prompt_box[0:4]).astype(np.int32)

                if -1 not in per_image_prompt_box:
                    left_top, right_bottom = (per_image_prompt_box[0],
                                              per_image_prompt_box[1]), (
                                                  per_image_prompt_box[2],
                                                  per_image_prompt_box[3])
                    cv2.rectangle(image_for_visual_prompt_box,
                                  left_top,
                                  right_bottom,
                                  color=prompt_box_color,
                                  thickness=2,
                                  lineType=cv2.LINE_AA)
                    text = f'prompt_box_{box_global_idx}'
                    text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
                    fill_right_bottom = (max(left_top[0] + text_size[0],
                                             right_bottom[0]),
                                         left_top[1] - text_size[1] - 3)
                    cv2.rectangle(image_for_visual_prompt_box,
                                  left_top,
                                  fill_right_bottom,
                                  color=prompt_box_color,
                                  thickness=-1,
                                  lineType=cv2.LINE_AA)
                    cv2.putText(image_for_visual_prompt_box,
                                text, (left_top[0], left_top[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color=(0, 0, 0),
                                thickness=1,
                                lineType=cv2.LINE_AA)
                box_global_idx += 1

        image_for_visual_prompt_mask = copy.deepcopy(visual_image).astype(
            np.uint8)

        for per_prompt_point in prompt_points:
            for per_point in per_prompt_point:
                point_label = per_point[2]
                if point_label == 1:
                    cv2.circle(image_for_visual_prompt_mask,
                               (int(per_point[0]), int(per_point[1])), 10,
                               positive_prompt_point_color, -1)
                elif point_label == 0:
                    cv2.circle(image_for_visual_prompt_mask,
                               (int(per_point[0]), int(per_point[1])), 10,
                               negative_prompt_point_color, -1)

        # draw all prompt masks on one image
        per_image_visual_prompt_mask = np.zeros(
            (image_for_visual_prompt_mask.shape[0],
             image_for_visual_prompt_mask.shape[1], 3))
        per_image_visual_prompt_contours = []
        for pmi, per_prompt_mask in enumerate(prompt_masks):
            prompt_mask_color = [
                int(np.random.choice(range(256))) for _ in range(3)
            ]
            per_prompt_mask = per_prompt_mask.astype(np.uint8)
            per_prompt_mask_nonzero = np.nonzero(per_prompt_mask == 1.)
            if len(per_prompt_mask_nonzero[0]) > 0:
                per_image_visual_prompt_mask[
                    per_prompt_mask_nonzero[0],
                    per_prompt_mask_nonzero[1]] = prompt_mask_color
            new_per_image_visual_prompt_mask = np.zeros(
                (image_for_visual_prompt_mask.shape[0],
                 image_for_visual_prompt_mask.shape[1]))
            if len(per_prompt_mask_nonzero[0]) > 0:
                new_per_image_visual_prompt_mask[
                    per_prompt_mask_nonzero[0],
                    per_prompt_mask_nonzero[1]] = 255
            contours, _ = cv2.findContours(
                new_per_image_visual_prompt_mask.astype(np.uint8),
                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            per_image_visual_prompt_contours.append(contours)
        per_image_visual_prompt_mask = per_image_visual_prompt_mask.astype(
            np.uint8)
        per_image_visual_prompt_mask = cv2.cvtColor(
            per_image_visual_prompt_mask, cv2.COLOR_RGB2BGR)
        all_classes_mask = np.nonzero(per_image_visual_prompt_mask != 0)
        if len(all_classes_mask[0]) > 0:
            per_image_visual_prompt_mask[
                all_classes_mask[0], all_classes_mask[1]] = cv2.addWeighted(
                    image_for_visual_prompt_mask[all_classes_mask[0],
                                                 all_classes_mask[1]], 0.5,
                    per_image_visual_prompt_mask[all_classes_mask[0],
                                                 all_classes_mask[1]], 1, 0)
        no_class_mask = np.nonzero(per_image_visual_prompt_mask == 0)
        if len(no_class_mask[0]) > 0:
            per_image_visual_prompt_mask[
                no_class_mask[0],
                no_class_mask[1]] = image_for_visual_prompt_mask[
                    no_class_mask[0], no_class_mask[1]]
        for contours in per_image_visual_prompt_contours:
            cv2.drawContours(per_image_visual_prompt_mask, contours, -1,
                             (255, 255, 255), 2)

        cv2.imencode('.jpg', image_for_visual_prompt_box)[1].tofile(
            os.path.join(
                temp_dir,
                f'idx_{count}_visual_image_with_prompt_point_box.jpg'))
        cv2.imencode('.jpg', per_image_visual_prompt_mask)[1].tofile(
            os.path.join(temp_dir,
                         f'idx_{count}_visual_image_with_prompt_mask.jpg'))

        ################################################################
        # print text prompt sample info
        ################################################################
        print('2121', text_sample['image_path'])
        print('2222', text_sample['image'].shape, text_sample['size'],
              text_sample['num_targets'], text_sample['sample_type'])
        print('2323', text_sample['image'].dtype, text_sample['size'].dtype)
        print('2424', len(text_sample['masks']),
              len(text_sample['prompt_texts']))
        print('2525', text_sample['pil_image'].size,
              text_sample['pil_image'].mode)

        for per_mask in text_sample['masks']:
            print('3333', per_mask.shape, per_mask.dtype)
        for per_text_description in text_sample['prompt_texts']:
            print('4444', per_text_description)

        ################################################################
        # visualize text prompt sample
        ################################################################
        text_image = np.ascontiguousarray(text_sample['image'], dtype=np.uint8)
        text_image = cv2.cvtColor(text_image, cv2.COLOR_RGB2BGR)
        # save text prompt pil_image (original resolution)
        text_pil_image = text_sample['pil_image']

        text_masks = text_sample['masks']
        text_masks_num = len(text_masks)

        # draw all text masks on one image
        image_for_text_mask = copy.deepcopy(text_image).astype(np.uint8)
        per_image_text_mask = np.zeros(
            (image_for_text_mask.shape[0], image_for_text_mask.shape[1], 3))
        per_image_text_contours = []
        for i in range(text_masks_num):
            per_mask = text_masks[i]
            text_mask_color = [
                int(np.random.choice(range(256))) for _ in range(3)
            ]
            per_mask = per_mask.astype(np.uint8)
            per_object_mask = np.nonzero(per_mask == 1.)
            if len(per_object_mask[0]) > 0:
                per_image_text_mask[per_object_mask[0],
                                    per_object_mask[1]] = text_mask_color
            new_per_image_text_mask = np.zeros(
                (image_for_text_mask.shape[0], image_for_text_mask.shape[1]))
            if len(per_object_mask[0]) > 0:
                new_per_image_text_mask[per_object_mask[0],
                                        per_object_mask[1]] = 255
            contours, _ = cv2.findContours(
                new_per_image_text_mask.astype(np.uint8), cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            per_image_text_contours.append(contours)
        per_image_text_mask = per_image_text_mask.astype(np.uint8)
        per_image_text_mask = cv2.cvtColor(per_image_text_mask,
                                           cv2.COLOR_RGB2BGR)
        all_classes_mask = np.nonzero(per_image_text_mask != 0)
        if len(all_classes_mask[0]) > 0:
            per_image_text_mask[all_classes_mask[0],
                                all_classes_mask[1]] = cv2.addWeighted(
                                    image_for_text_mask[all_classes_mask[0],
                                                        all_classes_mask[1]],
                                    0.5,
                                    per_image_text_mask[all_classes_mask[0],
                                                        all_classes_mask[1]],
                                    1, 0)
        no_class_mask = np.nonzero(per_image_text_mask == 0)
        if len(no_class_mask[0]) > 0:
            per_image_text_mask[no_class_mask[0],
                                no_class_mask[1]] = image_for_text_mask[
                                    no_class_mask[0], no_class_mask[1]]
        for contours in per_image_text_contours:
            cv2.drawContours(per_image_text_mask, contours, -1,
                             (255, 255, 255), 2)

        cv2.imencode('.jpg', image_for_text_mask)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}_text_image.jpg'))
        cv2.imencode('.jpg', per_image_text_mask)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}_text_image_with_mask.jpg'))
        text_pil_image.save(
            os.path.join(temp_dir, f'idx_{count}_text_pil_image.jpg'))

        if count < 10:
            count += 1
        else:
            break

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
            1: 1.0,
        },
        use_text_prompt_num_prob={
            1: 1.0,
        },
        use_description_type_prob={
            'absolute_position': 0.,
            'category': 0.,
            'phrase_description': 0.,
            'detail_description': 0.3,
            'absolute_detail_description': 0.4,
            'relative_detail_description': 0.3,
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
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        sample_type, input_images, sizes, pil_images, input_masks_list, num_targets_list = data[
            'sample_type'], data['image'], data['size'], data[
                'pil_image'], data['mask'], data['num_targets']
        print('6060', sample_type)
        print('6161', input_images.shape, sizes, len(pil_images),
              len(input_masks_list))
        print('6262', input_images.dtype, sizes.dtype)
        print('6363', num_targets_list)
        print('6464', f'prompt_text: {data["prompt_text"]}')
        print('6565', f'prompt_language: {data["prompt_language"]}')

        for per_image_masks in input_masks_list:
            print('7777', per_image_masks.shape)

        if sample_type == 'visual':
            input_prompt_masks_list = data['prompt_mask']
            for per_image_prompt_masks in input_prompt_masks_list:
                print('8888', per_image_prompt_masks.shape)

            temp_dir = './temp2'
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            for i, (per_image, per_image_masks_tensor,
                    per_image_prompt_masks_tensor, per_pil_image) in enumerate(
                        zip(input_images, input_masks_list,
                            input_prompt_masks_list, pil_images)):
                per_image = per_image.permute(1, 2, 0).cpu().numpy()
                per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
                per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

                per_image_masks = per_image_masks_tensor.cpu().numpy()
                per_image_masks_num = per_image_masks.shape[0]

                per_image_prompt_masks = per_image_prompt_masks_tensor.cpu(
                ).numpy()
                per_image_prompt_masks_num = per_image_prompt_masks.shape[0]

                # draw all masks on one image
                image_for_mask = copy.deepcopy(per_image).astype(np.uint8)
                per_image_draw_mask = np.zeros(
                    (image_for_mask.shape[0], image_for_mask.shape[1], 3))
                per_image_contours = []
                for mi in range(per_image_masks_num):
                    per_mask = per_image_masks[mi]
                    mask_color = [
                        int(np.random.choice(range(256))) for _ in range(3)
                    ]
                    per_mask_nonzero = np.nonzero(per_mask == 1.)
                    if len(per_mask_nonzero[0]) > 0:
                        per_image_draw_mask[per_mask_nonzero[0],
                                            per_mask_nonzero[1]] = mask_color
                    new_per_image_draw_mask = np.zeros(
                        (image_for_mask.shape[0], image_for_mask.shape[1]))
                    if len(per_mask_nonzero[0]) > 0:
                        new_per_image_draw_mask[per_mask_nonzero[0],
                                                per_mask_nonzero[1]] = 255
                    contours, _ = cv2.findContours(
                        new_per_image_draw_mask.astype(np.uint8),
                        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    per_image_contours.append(contours)
                per_image_draw_mask = per_image_draw_mask.astype(np.uint8)
                per_image_draw_mask = cv2.cvtColor(per_image_draw_mask,
                                                   cv2.COLOR_RGB2BGR)
                all_classes_mask = np.nonzero(per_image_draw_mask != 0)
                if len(all_classes_mask[0]) > 0:
                    per_image_draw_mask[
                        all_classes_mask[0],
                        all_classes_mask[1]] = cv2.addWeighted(
                            image_for_mask[all_classes_mask[0],
                                           all_classes_mask[1]], 0.5,
                            per_image_draw_mask[all_classes_mask[0],
                                                all_classes_mask[1]], 1, 0)
                no_class_mask = np.nonzero(per_image_draw_mask == 0)
                if len(no_class_mask[0]) > 0:
                    per_image_draw_mask[no_class_mask[0],
                                        no_class_mask[1]] = image_for_mask[
                                            no_class_mask[0], no_class_mask[1]]
                for contours in per_image_contours:
                    cv2.drawContours(per_image_draw_mask, contours, -1,
                                     (255, 255, 255), 2)

                cv2.imencode('.jpg', per_image)[1].tofile(
                    os.path.join(temp_dir,
                                 f'idx_{count}_{i}_visual_image.jpg'))
                cv2.imencode('.jpg', per_image_draw_mask)[1].tofile(
                    os.path.join(
                        temp_dir,
                        f'idx_{count}_{i}_visual_image_with_mask.jpg'))

                # draw all prompt masks on one image
                image_for_prompt_mask = copy.deepcopy(per_image).astype(
                    np.uint8)
                per_image_prompt_draw_mask = np.zeros(
                    (image_for_prompt_mask.shape[0],
                     image_for_prompt_mask.shape[1], 3))
                per_image_prompt_contours = []
                for pmi in range(per_image_prompt_masks_num):
                    per_prompt_mask = per_image_prompt_masks[pmi]
                    prompt_mask_color = [
                        int(np.random.choice(range(256))) for _ in range(3)
                    ]
                    per_prompt_mask_nonzero = np.nonzero(per_prompt_mask == 1.)
                    if len(per_prompt_mask_nonzero[0]) > 0:
                        per_image_prompt_draw_mask[
                            per_prompt_mask_nonzero[0],
                            per_prompt_mask_nonzero[1]] = prompt_mask_color
                    new_per_image_prompt_draw_mask = np.zeros(
                        (image_for_prompt_mask.shape[0],
                         image_for_prompt_mask.shape[1]))
                    if len(per_prompt_mask_nonzero[0]) > 0:
                        new_per_image_prompt_draw_mask[
                            per_prompt_mask_nonzero[0],
                            per_prompt_mask_nonzero[1]] = 255
                    contours, _ = cv2.findContours(
                        new_per_image_prompt_draw_mask.astype(np.uint8),
                        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    per_image_prompt_contours.append(contours)
                per_image_prompt_draw_mask = per_image_prompt_draw_mask.astype(
                    np.uint8)
                per_image_prompt_draw_mask = cv2.cvtColor(
                    per_image_prompt_draw_mask, cv2.COLOR_RGB2BGR)
                all_classes_mask = np.nonzero(per_image_prompt_draw_mask != 0)
                if len(all_classes_mask[0]) > 0:
                    per_image_prompt_draw_mask[
                        all_classes_mask[0],
                        all_classes_mask[1]] = cv2.addWeighted(
                            image_for_prompt_mask[all_classes_mask[0],
                                                  all_classes_mask[1]], 0.5,
                            per_image_prompt_draw_mask[all_classes_mask[0],
                                                       all_classes_mask[1]], 1,
                            0)
                no_class_mask = np.nonzero(per_image_prompt_draw_mask == 0)
                if len(no_class_mask[0]) > 0:
                    per_image_prompt_draw_mask[
                        no_class_mask[0],
                        no_class_mask[1]] = image_for_prompt_mask[
                            no_class_mask[0], no_class_mask[1]]
                for contours in per_image_prompt_contours:
                    cv2.drawContours(per_image_prompt_draw_mask, contours, -1,
                                     (255, 255, 255), 2)

                cv2.imencode('.jpg', per_image_prompt_draw_mask)[1].tofile(
                    os.path.join(
                        temp_dir,
                        f'idx_{count}_{i}_visual_image_with_prompt_mask.jpg'))
                per_pil_image.save(
                    os.path.join(temp_dir,
                                 f'idx_{count}_{i}_visual_pil_image.jpg'))

        elif sample_type == 'text':
            input_prompt_texts = data['prompt_text']
            input_prompt_languages = data['prompt_language']

            print('9393', f'prompt_texts({len(input_prompt_texts)}):',
                  input_prompt_texts)
            print('9494', f'prompt_languages({len(input_prompt_languages)}):',
                  input_prompt_languages)

            temp_dir = './temp2'
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            for i, (per_image, per_image_masks_tensor, per_prompt_text,
                    per_prompt_language, per_pil_image) in enumerate(
                        zip(input_images, input_masks_list, input_prompt_texts,
                            input_prompt_languages, pil_images)):
                per_image = per_image.permute(1, 2, 0).cpu().numpy()
                per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
                per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

                per_image_masks = per_image_masks_tensor.cpu().numpy()
                per_image_masks_num = per_image_masks.shape[0]

                print('9595', f'prompt_text: {per_prompt_text}')
                print('9696', f'prompt_language: {per_prompt_language}')

                # draw all masks on one image
                image_for_mask = copy.deepcopy(per_image).astype(np.uint8)
                per_image_draw_mask = np.zeros(
                    (image_for_mask.shape[0], image_for_mask.shape[1], 3))
                per_image_contours = []
                for mi in range(per_image_masks_num):
                    per_mask = per_image_masks[mi]
                    mask_color = [
                        int(np.random.choice(range(256))) for _ in range(3)
                    ]
                    per_mask_nonzero = np.nonzero(per_mask == 1.)
                    if len(per_mask_nonzero[0]) > 0:
                        per_image_draw_mask[per_mask_nonzero[0],
                                            per_mask_nonzero[1]] = mask_color
                    new_per_image_draw_mask = np.zeros(
                        (image_for_mask.shape[0], image_for_mask.shape[1]))
                    if len(per_mask_nonzero[0]) > 0:
                        new_per_image_draw_mask[per_mask_nonzero[0],
                                                per_mask_nonzero[1]] = 255
                    contours, _ = cv2.findContours(
                        new_per_image_draw_mask.astype(np.uint8),
                        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    per_image_contours.append(contours)
                per_image_draw_mask = per_image_draw_mask.astype(np.uint8)
                per_image_draw_mask = cv2.cvtColor(per_image_draw_mask,
                                                   cv2.COLOR_RGB2BGR)
                all_classes_mask = np.nonzero(per_image_draw_mask != 0)
                if len(all_classes_mask[0]) > 0:
                    per_image_draw_mask[
                        all_classes_mask[0],
                        all_classes_mask[1]] = cv2.addWeighted(
                            image_for_mask[all_classes_mask[0],
                                           all_classes_mask[1]], 0.5,
                            per_image_draw_mask[all_classes_mask[0],
                                                all_classes_mask[1]], 1, 0)
                no_class_mask = np.nonzero(per_image_draw_mask == 0)
                if len(no_class_mask[0]) > 0:
                    per_image_draw_mask[no_class_mask[0],
                                        no_class_mask[1]] = image_for_mask[
                                            no_class_mask[0], no_class_mask[1]]
                for contours in per_image_contours:
                    cv2.drawContours(per_image_draw_mask, contours, -1,
                                     (255, 255, 255), 2)

                cv2.imencode('.jpg', per_image)[1].tofile(
                    os.path.join(temp_dir, f'idx_{count}_{i}_text_image.jpg'))
                cv2.imencode('.jpg', per_image_draw_mask)[1].tofile(
                    os.path.join(temp_dir,
                                 f'idx_{count}_{i}_text_image_with_mask.jpg'))
                per_pil_image.save(
                    os.path.join(temp_dir,
                                 f'idx_{count}_{i}_text_pil_image.jpg'))

        if count < 10:
            count += 1
        else:
            break
