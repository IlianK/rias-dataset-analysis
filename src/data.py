import re
from utils import _paths, _params, _dataset_names
import os
import numpy as np
import json
import shutil
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from IPython.display import display


"""
DATA PREPERATION
"""


# PREPROCESS MASK
def preprocess_mask(json_path, image_shape):
    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for item in data:
        x1, y1, x2, y2 = item['bounding_box']
        height, width = image_shape[:2]

        x1, y1, x2, y2 = [int(coord * size) for coord, size
                          in zip([x1, y1, x2, y2],
                                 [width, height, width, height])]

        mask[y1:y2, x1:x2] = 255

    mask = Image.fromarray(mask)
    mask = mask.resize(_params.target_size, resample=Image.NEAREST)
    mask = np.array(mask).astype('float32') / 255.

    return mask[..., np.newaxis]


# PREPROCESS IMAGE WITH MASK
def preprocess_image_with_mask(image_path, json_path):
    img = Image.open(image_path).convert("RGB")
    mask = preprocess_mask(json_path, img.size)

    if _params.target_size is not None:
        img.thumbnail(_params.target_size)
        padding = (
            (_params.target_size[0] - img.size[0]) // 2,
            (_params.target_size[1] - img.size[1]) // 2
        )

        border_color = img.getpixel((0, 0))
        padded_img = Image.new("RGB", _params.target_size, border_color)
        padded_img.paste(img, padding)

        img = padded_img

    img = np.array(img).astype('float32') / 255.
    img_with_mask = np.dstack([img, mask])

    return img_with_mask


# SAVE PREPROCESSED DATASETS
def save_datasets_prepared(dataset_names):
    for dataset_name in dataset_names:
        data_dir = os.path.join(_paths.datasets_prepared, dataset_name)
        os.makedirs(data_dir, exist_ok=True)

        original_data_dir = os.path.join(_paths.datasets, dataset_name)
        data_files = [file for file in os.listdir(original_data_dir)
                      if file not in ['summary.json'] and '_mask' not in file]

        for i in range(len(data_files) // 2):
            json_path = os.path.join(original_data_dir, f'{i}.json')
            image_path = os.path.join(original_data_dir, f'{i}.png')

            img_with_mask = preprocess_image_with_mask(image_path, json_path)
            save_path = os.path.join(data_dir,
                                     f'preprocessed_{dataset_name}_{i}.npy')

            np.save(save_path, img_with_mask)
            print("Saved preprocessed img at {} with shape: {}".format(
                save_path, img_with_mask.shape))


# MOVE FILES TO DIRECTORY
def move_files_to_directory(file_paths, dir, sub_dir):
    for file_path in file_paths:
        des_path = os.path.join(dir, sub_dir, os.path.basename(file_path))
        shutil.move(file_path, des_path)


# SPLIT DATASETS INTO TRAIN VAL TEST
def split_datasets_into_train_val_test(dataset_names):
    for dataset_name in dataset_names:
        dataset_path = os.path.join(_paths.datasets_prepared, dataset_name)

        subsets = ['train', 'val', 'test']
        for subset in subsets:
            os.makedirs(os.path.join(dataset_path, subset), exist_ok=True)

        all_data = [os.path.join(dataset_path, f) for f
                    in os.listdir(os.path.join(dataset_path))
                    if f.endswith('.npy')]

        train_data, test_data = train_test_split(all_data,
                                                 test_size=_params.test_size,
                                                 random_state=42,
                                                 shuffle=_params.shuffle)
        train_data, val_data = train_test_split(train_data,
                                                test_size=_params.val_size,
                                                random_state=42,
                                                shuffle=_params.shuffle)

        move_files_to_directory(train_data, dataset_path, 'train')
        move_files_to_directory(val_data, dataset_path, 'val')
        move_files_to_directory(test_data, dataset_path, 'test')


# DISPLAY IMG AND MASK REBUILD
def display_img_mask(img, mask, title_prefix=''):
    plt.subplot(1, 2, 1)
    plt.imshow((img * 255).astype(np.uint8))
    plt.title(f'{title_prefix} Image')

    plt.subplot(1, 2, 2)
    plt.imshow((mask * 255).astype(np.uint8), cmap='gray')
    plt.title(f'{title_prefix} Mask')

    plt.show()


# GET RANDOM NPY OR ORIGINAL TUPLE PATH (PNG, JSON)
def get_random_file(dataset_name, is_prepared, img_type='test'):
    base_dir = _paths.datasets_prepared_whole if is_prepared else _paths.datasets
    dataset_dir = os.path.join(base_dir, dataset_name, img_type) if is_prepared else os.path.join(base_dir, dataset_name)

    if is_prepared:
        all_files = os.listdir(dataset_dir)
        npy_files = [f for f in all_files if f.endswith('.npy')]
        random_file = random.choice(npy_files)
        random_file_path = os.path.join(dataset_dir, random_file)
        random_data = np.load(random_file_path)
        print("Random npy file path:", random_file_path)
        print("Dataset:", dataset_name)
        return random_data
    
    else:
        png_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
        random_png = random.choice(png_files)
        random_png_path = os.path.join(dataset_dir, random_png)
        json_file = random_png_path.replace('.png', '.json')
        json_file_path = os.path.join(dataset_dir, json_file)
        print("Random PNG file path:", random_png_path)
        print("Corresponding JSON file path:", json_file_path)
        print("Dataset:", dataset_name)
        return random_png_path, json_file_path


# TEST PREPROCESSING MASK AND IMAGE REBUILD
def test_preprocess_image(dataset_name):
    random_png_path, random_json_path = get_random_file(dataset_name, is_prepared=False)  
    original_img = Image.open(random_png_path)
    
    img_with_mask = preprocess_image_with_mask(random_png_path, random_json_path)
    rgb_img = img_with_mask[..., :3]
    mask_channel = img_with_mask[..., 3]
    
    original_img.show()
    display_img_mask(rgb_img, mask_channel, 'Preprocessed')


# VERIFY SAVED PREPROCESSED MASK AND IMAGE
def verify_preprocess_image(dataset_name, img_type):
    random_data = get_random_file(dataset_name, is_prepared=True, img_type=img_type)
    rgb_img = random_data[..., :3]
    mask_channel = random_data[..., 3]
    display_img_mask(rgb_img, mask_channel)
    

'''
DATA COMPRESSION AND STRUCTURING
'''


# RETRY PREPROCESSING FOR CORRUPTED
def retry_preprocessing_for_corrupted(dataset_name, img_type, num):
    image_path = os.path.join(_paths.datasets, f'{num}.png')
    json_path = os.path.join(_paths.datasets, f'{num}.json')

    img_with_mask = preprocess_image_with_mask(image_path, json_path)
    temp_dest = os.path.join(_paths.datasets_prepared_whole,
                             dataset_name, img_type,
                             f'preprocessed_{dataset_name}_{num}')
    np.save(temp_dest, img_with_mask)


# GET RANDOM FILES FOR RAR
def create_random_rar_files(dataset_name, img_type, n, output_path):
    path = os.path.join(_paths.datasets_prepared, dataset_name, img_type)
    temp_dir = os.path.join(_paths.datasets_prepared, 'temp_random_rar')

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    files_in_dataset = os.listdir(path)
    n = min(n, len(files_in_dataset))
    selected_files = random.sample(files_in_dataset, n)

    try:
        for file in selected_files:
            file_path = os.path.join(path, file)
            shutil.copy(file_path, temp_dir)

        shutil.make_archive(output_path[:-4], 'zip', temp_dir)
        os.rename(output_path[:-4] + '.zip', output_path)

        print(f'Rar file created at: {output_path}')

    finally:
        shutil.rmtree(temp_dir)


# CREATE RARS OF PREPROCESSED DATA
def create_rars(datasets, train_n, val_n):
    for dataset_name in datasets:
        for data_type in ['train', 'val']:
            output_path = os.path.join(_paths.datasets_prepared,
                                       'rars', dataset_name, data_type,
                                       f'{data_type}.rar')
            create_random_rar_files(dataset_name, data_type, train_n
                                    if data_type == 'train' else val_n,
                                    output_path)


# MOVE FILES WITH PATTERN
def move_files_with_pattern(src_dir, dest_dir, pattern, amount_to_move):
    os.makedirs(dest_dir, exist_ok=True)
    files = os.listdir(src_dir)

    matching_files = [file for file in files if re.match(pattern, file)]
    random.shuffle(matching_files)

    for file in matching_files[:amount_to_move]:
        shutil.move(os.path.join(src_dir, file), os.path.join(dest_dir, file))


# RENAME FILE
def rename_file(old_path, new_path):
    try:
        os.rename(old_path, new_path)
        print(f"File '{old_path}' renamed to '{new_path}'.")
    except Exception as e:
        print(f"Error renaming file: {e}")


# RENAME MODEL
def rename_model(models_folder, previous_name, new_name):
    old_model_path = os.path.join(_paths.models, models_folder, previous_name)
    new_model_path = os.path.join(_paths.models, models_folder, new_name)

    file_mappings = {
        'autoencoder_architecture.json': 'autoencoder',
        'autoencoder_weights.h5': 'autoencoder',
        'encoder_architecture.json': 'encoder',
        'encoder_weights.h5': 'encoder',
        'checkpoint.h5': 'checkpoints',
        'history.json': ''
    }

    for file_name, subfolder in file_mappings.items():
        base_path = os.path.join(old_model_path, subfolder)
        old_file_path = os.path.join(base_path, f'{previous_name}_{file_name}')
        new_file_path = os.path.join(base_path, f'{new_name}_{file_name}')
        rename_file(old_file_path, new_file_path)

    rename_file(old_model_path, new_model_path)


# COPY FILE
def copy_file(source_path, destination_path):
    try:
        shutil.copy2(source_path, destination_path)
        print(f"File copied from {source_path} to {destination_path}")
    except FileNotFoundError:
        print("Source file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# COPY HISTORIES
def copy_history_files(model_info, base_path, dest_dir):
    for model_dir, models in model_info.items():
        for model_name in models:
            model_folder = os.path.join(base_path, model_dir, model_name)
            history_file = os.path.join(model_folder, f'{model_name}_history.json')
            if os.path.exists(history_file):
                shutil.copy(history_file, dest_dir)


# VERIFY IF FILE EXISTS
def verify_file_exists(file_path):
    if os.path.exists(file_path):
        print(f"File {file_path} exists")
    else:
        print(f"File {file_path} doesn't exists")
