import os
import re
import json
import shutil
import numpy as np
import cv2
import collections
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from pycocotools import mask as mask_utils

# ====================== 原始数据集配置 ======================
DATASET_ROOT = '/root/autodl-tmp/interactive_segmentation_dataset'
# ✅ 要处理的子集列表, list 形式便于灵活指定
SUBSET_NAME_LIST = [
    'sa_000000',
]
SPLIT_NAME = 'train'

# ====================== 002 脚本输出配置 ======================
CAPTION_SAVE_ROOT = '/root/autodl-tmp/interactive_segmentation_dataset_captions'
GLOBAL_CAPTION_SUFFIX = '_global_caption.json'
OUTPUT_JSON_SUFFIX = '_deepseek_flash_output.json'

DESCRIPTION_KEY_LIST = [
    'absolute_position',
    'category',
    'phrase_description',
    'detail_description',
    'absolute_detail_description',
    'relative_detail_description',
]


def parse_output_text(per_output_text):
    per_parts = per_output_text.split('---ENGLISH_TRANSLATION---')
    per_chinese_part = per_parts[0].strip()
    per_english_part = per_parts[1].strip()

    per_chinese_lines = [
        line.strip() for line in per_chinese_part.split('\n') if line.strip()
    ]
    per_english_lines = [
        line.strip() for line in per_english_part.split('\n') if line.strip()
    ]

    per_chinese_desc_dict = {}
    for per_idx, per_line in enumerate(per_chinese_lines):
        if per_idx >= len(DESCRIPTION_KEY_LIST):
            break
        per_line = re.sub(r'^\d+[、．.]\s*', '', per_line)
        per_chinese_desc_dict[DESCRIPTION_KEY_LIST[per_idx]] = per_line

    per_english_desc_dict = {}
    for per_idx, per_line in enumerate(per_english_lines):
        if per_idx >= len(DESCRIPTION_KEY_LIST):
            break
        per_line = re.sub(r'^\d+[.、．]\s*', '', per_line)
        per_english_desc_dict[DESCRIPTION_KEY_LIST[per_idx]] = per_line

    # validate no empty string values
    for per_key in DESCRIPTION_KEY_LIST:
        if per_key not in per_chinese_desc_dict or not per_chinese_desc_dict[
                per_key]:
            return None
        if per_key not in per_english_desc_dict or not per_english_desc_dict[
                per_key]:
            return None

    return per_chinese_desc_dict, per_english_desc_dict


def validate_single_caption_file(args):
    """
    校验 002 脚本输出的单个 caption json 是否可用
    文件名形如: {image_name}_{ann_idx}_deepseek_flash_output.json
    """
    per_caption_file_name, per_caption_dir, per_split_dir = args

    per_caption_json_path = os.path.join(per_caption_dir,
                                         per_caption_file_name)

    # parse image name and mask index from caption file name
    per_name_without_suffix = per_caption_file_name[:-len(OUTPUT_JSON_SUFFIX)]
    per_parts = per_name_without_suffix.rsplit('_', 1)
    if len(per_parts) != 2:
        return None

    per_image_name = per_parts[0]
    try:
        per_ann_idx = int(per_parts[1])
    except ValueError:
        return None

    # check source image and source annotation json exist
    per_image_path = os.path.join(per_split_dir, per_image_name + '.jpg')
    per_source_json_path = os.path.join(per_split_dir,
                                        per_image_name + '.json')
    if not os.path.exists(per_image_path) or not os.path.exists(
            per_source_json_path):
        return None

    # check annotation json status == "ok"
    try:
        with open(per_caption_json_path, 'r', encoding='utf-8') as f:
            per_anno_data = json.load(f)
        if per_anno_data.get('status') != 'ok':
            print(f'{per_caption_json_path} status not ok!')
            return None
        if not per_anno_data.get('output_text'):
            print(f'{per_caption_json_path} empty output_text!')
            return None
    except Exception:
        print(f'error when {OUTPUT_JSON_SUFFIX}!')
        return None

    return (per_image_name, per_ann_idx, per_caption_json_path, per_image_path,
            per_source_json_path)


def process_single_group(args, save_dataset_path):
    per_image_name, per_ann_idx_list, per_caption_path_list, per_image_src_path, per_source_json_path, per_save_subset_name = args

    per_save_folder_path = os.path.join(save_dataset_path,
                                        per_save_subset_name, 'train',
                                        per_image_name)
    os.makedirs(per_save_folder_path, exist_ok=True)

    # save image file from the shared source dataset image
    per_image_save_path = os.path.join(per_save_folder_path,
                                       per_image_name + '.jpg')
    per_image = cv2.imdecode(np.fromfile(per_image_src_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
    if per_image is None:
        print(f'can not read image: {per_image_src_path}')
        shutil.rmtree(per_save_folder_path, ignore_errors=True)
        return
    per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]
    cv2.imencode('.jpg', per_image)[1].tofile(per_image_save_path)

    # read source annotation json once, all masks of this image share it
    try:
        with open(per_source_json_path, 'r', encoding='utf-8') as f:
            per_source_data = json.load(f)
        per_source_annotation_list = per_source_data['annotations']
    except Exception:
        print(f'can not read source json: {per_source_json_path}')
        shutil.rmtree(per_save_folder_path, ignore_errors=True)
        return

    # save mask files and collect annotation info
    per_annotation_dict = {}
    for per_ann_idx, per_caption_path in zip(per_ann_idx_list,
                                             per_caption_path_list):
        # decode mask from source RLE annotation (keep _index suffix)
        per_mask_save_name = f'{per_image_name}_{per_ann_idx}.png'
        per_mask_save_path = os.path.join(per_save_folder_path,
                                          per_mask_save_name)

        if per_ann_idx >= len(per_source_annotation_list):
            print(f'ann_idx {per_ann_idx} out of range!')
            shutil.rmtree(per_save_folder_path, ignore_errors=True)
            return

        per_mask = mask_utils.decode(
            per_source_annotation_list[per_ann_idx]['segmentation'])
        if per_mask is None:
            print(f'can not decode mask: {per_source_json_path}')
            shutil.rmtree(per_save_folder_path, ignore_errors=True)
            return
        if per_mask.ndim == 3:
            per_mask = per_mask[:, :, 0]
        per_mask = per_mask.astype(np.uint8)
        per_mask[per_mask > 0] = 255

        # check mask size matches image size
        per_mask_h, per_mask_w = per_mask.shape[0], per_mask.shape[1]
        if per_mask_h != per_image_h or per_mask_w != per_image_w:
            print(f'mask_h != image_h  or mask_w != image_w!')
            shutil.rmtree(per_save_folder_path, ignore_errors=True)
            return

        # check mask foreground area ratio
        per_mask_area = np.sum(per_mask > 0)
        per_image_area = per_image_h * per_image_w
        per_mask_ratio = per_mask_area / per_image_area
        if per_mask_ratio < 0.0001 or per_mask_ratio > 0.9:
            print(f'mask area ratio {per_mask_ratio:.6f} out of range!')
            shutil.rmtree(per_save_folder_path, ignore_errors=True)
            return

        # compute mask bbox [x_min, y_min, w, h] and mask area
        per_mask_nonzero = np.nonzero(per_mask > 0)  # (rows, cols)
        if len(per_mask_nonzero[0]) > 0:
            per_y_min = int(np.min(per_mask_nonzero[0]))
            per_y_max = int(np.max(per_mask_nonzero[0]))
            per_x_min = int(np.min(per_mask_nonzero[1]))
            per_x_max = int(np.max(per_mask_nonzero[1]))
            per_mask_box = [
                per_x_min, per_y_min, per_x_max - per_x_min,
                per_y_max - per_y_min
            ]
        else:
            per_mask_box = [0, 0, 0, 0]
        per_mask_area_pixels = int(np.sum(per_mask > 0))

        cv2.imencode('.png', per_mask)[1].tofile(per_mask_save_path)

        # read annotation json generated by the 002 script
        with open(per_caption_path, 'r', encoding='utf-8') as f:
            per_anno_data = json.load(f)

        per_output_text = per_anno_data['output_text']
        per_parse_result = parse_output_text(per_output_text)
        if per_parse_result is None:
            print(f'empty string values in per_output_text!')
            shutil.rmtree(per_save_folder_path, ignore_errors=True)
            return
        per_chinese_desc_dict, per_english_desc_dict = per_parse_result

        per_annotation_dict[per_mask_save_name] = {
            'english': per_english_desc_dict,
            'chinese': per_chinese_desc_dict,
            'mask_box': per_mask_box,
            'mask_area': per_mask_area_pixels,
            'mask_h': per_mask_h,
            'mask_w': per_mask_w,
        }

    # save integrated annotation json
    per_anno_save_path = os.path.join(per_save_folder_path,
                                      per_image_name + '.json')
    with open(per_anno_save_path, 'w', encoding='utf-8') as f:
        json.dump(per_annotation_dict, f, ensure_ascii=False)

    # check if save folder is empty
    if os.path.isdir(
            per_save_folder_path) and not os.listdir(per_save_folder_path):
        print(f'empty save folder: {per_save_folder_path}')
        shutil.rmtree(per_save_folder_path, ignore_errors=True)
        return


def preprocess_image(root_dataset_path, subset_name_list, save_dataset_path):
    os.makedirs(save_dataset_path, exist_ok=True)

    # group by image name globally across all subsets, since the same image may
    # have its masks split into different source subsets (e.g. sa1b_8 holds
    # sa_2386108_0 while sa1b_8_1 holds sa_2386108_1). grouping per subset would
    # save the same image name into two different save subsets, each with a
    # partial json, which breaks the global uniqueness of image name.
    all_image_group_dict = collections.OrderedDict()
    # image name -> the first source subset it appears in, used to decide which
    # save subset this image belongs to. the save subset name is exactly the
    # original subset name, so an image whose masks are spread over several
    # source subsets is saved once, into the subset it first appears in.
    image_name_owner_subset_dict = collections.OrderedDict()

    for per_subset_name in tqdm(subset_name_list):
        per_caption_dir = os.path.join(root_dataset_path, per_subset_name,
                                       SPLIT_NAME)
        per_split_dir = os.path.join(DATASET_ROOT, per_subset_name, SPLIT_NAME)
        if not os.path.exists(per_caption_dir):
            print(f'caption dir not found: {per_caption_dir}')
            continue

        per_caption_file_name_list = sorted([
            per_file_name for per_file_name in os.listdir(per_caption_dir)
            if per_file_name.endswith(OUTPUT_JSON_SUFFIX)
        ])

        # validate and collect valid caption files (parallel)
        per_validate_args_list = [
            (per_caption_file_name, per_caption_dir, per_split_dir)
            for per_caption_file_name in per_caption_file_name_list
        ]
        with Pool(processes=32) as pool:
            per_validate_result_list = list(
                tqdm(pool.imap(validate_single_caption_file,
                               per_validate_args_list,
                               chunksize=256),
                     total=len(per_validate_args_list),
                     desc=f'scanning {per_subset_name}'))

        per_valid_result_list = [
            per_result for per_result in per_validate_result_list
            if per_result is not None
        ]

        print(f'{per_subset_name}: {len(per_valid_result_list)} valid masks')

        # group by image name into the global dict, so masks of the same image
        # coming from different source subsets are merged into one group
        per_subset_image_name_set = set()
        for per_result in per_valid_result_list:
            (per_image_name, per_index, per_caption_json_path, per_image_path,
             per_source_json_path) = per_result

            if per_image_name not in all_image_group_dict:
                all_image_group_dict[per_image_name] = []
            all_image_group_dict[per_image_name].append(
                (per_index, per_caption_json_path, per_image_path,
                 per_source_json_path))

            if per_image_name not in image_name_owner_subset_dict:
                image_name_owner_subset_dict[per_image_name] = per_subset_name

            per_subset_image_name_set.add(per_image_name)

        print(f'{per_subset_name}: '
              f'{len(per_subset_image_name_set)} image groups')

    # sort each group by index
    for per_image_name in all_image_group_dict:
        all_image_group_dict[per_image_name].sort(key=lambda x: x[0])

    print(f'total image groups after global grouping: '
          f'{len(all_image_group_dict)}')

    # sanity check: mask index must be unique inside each group, otherwise two
    # source folders would map to the same mask save name and overwrite each
    # other silently
    for per_image_name, per_group_items in all_image_group_dict.items():
        per_index_list = [item[0] for item in per_group_items]
        if len(set(per_index_list)) != len(per_index_list):
            raise RuntimeError(
                f'duplicated mask index in image group {per_image_name}: '
                f'{[item[1] for item in per_group_items]}')

    # bucket image names by their owner subset. one source subset is saved as
    # one single save subset without any sharding, so the save subset keeps the
    # original subset size and the original subset name.
    subset_to_image_name_list = collections.OrderedDict(
        (per_subset_name, []) for per_subset_name in subset_name_list)
    for per_image_name in all_image_group_dict:
        per_owner_subset_name = image_name_owner_subset_dict[per_image_name]
        subset_to_image_name_list[per_owner_subset_name].append(per_image_name)

    all_group_args_list = []
    for per_subset_name, per_group_key_list in subset_to_image_name_list.items(
    ):
        per_save_subset_name = per_subset_name

        print(f'{per_save_subset_name}: '
              f'{len(per_group_key_list)} image groups to save')

        for per_image_name in per_group_key_list:
            per_group_items = all_image_group_dict[per_image_name]

            per_ann_idx_list = [item[0] for item in per_group_items]
            per_caption_path_list = [item[1] for item in per_group_items]
            # all masks of one image share the same source image and json
            per_image_src_path = per_group_items[0][2]
            per_source_json_path = per_group_items[0][3]

            all_group_args_list.append((
                per_image_name,
                per_ann_idx_list,
                per_caption_path_list,
                per_image_src_path,
                per_source_json_path,
                per_save_subset_name,
            ))

    # sanity check: each image name must be saved exactly once, so that the
    # dataset class can safely use the image file name as a unique key
    all_save_image_name_list = [item[0] for item in all_group_args_list]
    if len(set(all_save_image_name_list)) != len(all_save_image_name_list):
        per_duplicated_name_list = [
            per_name for per_name, per_count in collections.Counter(
                all_save_image_name_list).items() if per_count > 1
        ]
        raise RuntimeError(
            f'{len(per_duplicated_name_list)} duplicated image names found: '
            f'{per_duplicated_name_list[0:10]}')

    print(f'total groups to process: {len(all_group_args_list)}')

    worker_fn = partial(process_single_group,
                        save_dataset_path=save_dataset_path)
    with Pool(processes=32) as pool:
        list(
            tqdm(pool.imap_unordered(worker_fn,
                                     all_group_args_list,
                                     chunksize=64),
                 total=len(all_group_args_list)))


if __name__ == '__main__':
    # ✅ 002 脚本的输出目录作为输入
    root_dataset_path = CAPTION_SAVE_ROOT
    subset_name_list = SUBSET_NAME_LIST
    save_dataset_path = r'/root/autodl-tmp/text_prompt_segmentation_dataset'
    preprocess_image(root_dataset_path, subset_name_list, save_dataset_path)
