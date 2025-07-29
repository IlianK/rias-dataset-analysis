import json
from matplotlib import pyplot as plt
import pandas as pd
import yaml
from utils import _paths, _dataset_names
import model_train
import os
import numpy as np
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import gc

'''
EVALUATION
'''


# LOAD ORIGINAL TEST IMAGES FOR EVALUATION
def load_images(dataset_name, image_type, amount, selection_option='random'):
    base_path = os.path.join(_paths.datasets_prepared_whole, dataset_name)
    images_path = os.path.join(base_path, image_type)
    image_files = os.listdir(images_path)

    if selection_option == 'random':
        amount = min(amount, len(image_files))
        selected_files = np.random.choice(image_files, amount, replace=False)

    elif selection_option == 'sequential':
        selected_files = image_files[:amount]

    elif selection_option == 'specific':
        pattern = f"preprocessed_{dataset_name}_{amount}.npy"
        selected_files = [file for file in image_files if pattern in file]

    images = []
    for file_name in selected_files:
        image_path = os.path.join(images_path, file_name)
        image_data = np.load(image_path)
        images.append(image_data)

    images = np.array(images)

    return images


# LOAD DECODED TEST IMAGES FOR EVALUATION
def load_decoded_images(dataset_name, image_type, models_folder, model_name):
    decoded_path = os.path.join(
        _paths.saves, 'decoded', models_folder, model_name,
        f'{model_name}_{dataset_name}_decoded_{image_type}.npy'
    )
    print(f'Try to load from path: {decoded_path}')

    try:
        decoded_image = np.load(decoded_path)
        return decoded_image

    except FileNotFoundError:
        print(f"Decoded {image_type} image batch not found for "
              f"{dataset_name} decoded by model {model_name}")
        return None


# APPEND ENTRY TO METRIC YAML
def append_entry_to_metric_yaml(model_name, dataset_name, metric_scores):
    try:
        yaml_file_path = os.path.join(_paths.yaml_files.all_scores)
        yaml = ruamel.yaml.YAML()
        try:
            with open(yaml_file_path, 'r') as file:
                data = yaml.load(file) or CommentedMap()
        except FileNotFoundError:
            data = CommentedMap()

        if 'models' not in data:
            data['models'] = CommentedMap()

        if model_name not in data['models']:
            data['models'][model_name] = CommentedMap()

        if dataset_name not in data['models'][model_name]:
            data['models'][model_name][dataset_name] = CommentedMap()

        for metric_name, metric_score in metric_scores.items():
            metric_score_str = str(metric_score)
            data['models'][model_name][dataset_name][
                metric_name.lower()] = metric_score_str

        with open(yaml_file_path, "w") as file:
            yaml.dump(data, file)

        print('Added yaml entry:')
        print(f'-- Model: {model_name}, Dataset: {dataset_name} ',
              f'Metrics: {metric_scores}')

    except Exception as e:
        print(f'Error appending entry to YAML: {e}')


'''
DECODE IMAGES
'''


# LOAD IMAGES TO DECODE FROM FOLDERS
def load_images_from_folders(dataset_name, img_type, amount):
    folder_suffixes = ['', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9']
    total_images_count = 0
    original_images_list = []

    for suffix in folder_suffixes:
        if total_images_count >= amount:
            break

        folder_type = f"{img_type}{suffix}"
        images = load_images(dataset_name, folder_type, amount, 'sequential') 

        remaining_amount = amount - total_images_count
        images_to_add = images[:remaining_amount]

        original_images_list.append(images_to_add)
        total_images_count += len(images_to_add)

    all_original_images = np.concatenate(original_images_list, axis=0)
    gc.collect()
    return all_original_images


# DECODING IMAGES
def decode_images(images, autoencoder, batch_size=1):
    decoded_images = np.empty_like(images)

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        decoded_batch = autoencoder.predict(batch)
        decoded_images[i:i + batch_size] = decoded_batch
    gc.collect()
    return decoded_images


# SAVE DECODED IMAGES
def save_decoded_images(decoded_images, dataset_name, img_type, models_folder, model_name):
    decoded_model_dir = os.path.join(_paths.decoded, models_folder, model_name)
    os.makedirs(decoded_model_dir, exist_ok=True)

    file_name = f'{model_name}_{dataset_name}_decoded_{img_type}.npy'
    decoded_path = os.path.join(decoded_model_dir, file_name)

    np.save(decoded_path, decoded_images)


# PROCESS AND SAVE IMAGES
def process_images_and_collect(dataset_names, amount, img_type, models_folder, model_name, autoencoder):
    all_datasets = {}  
    for dataset_name in dataset_names:
        original_images = load_images_from_folders(dataset_name, img_type, amount)
        gc.collect()

        decoded_images = decode_images(original_images, autoencoder)
        gc.collect()

        save_decoded_images(decoded_images, dataset_name, img_type,
                            models_folder, model_name)
        gc.collect()

        all_datasets[dataset_name] = (original_images, decoded_images)

        print(f'Processed and saved {amount} images for dataset ',
              f'{dataset_name} using model {model_name}.')

    return all_datasets


# VERIFY DECODED
def verify_decoded(models_folder, model_names):
    for model_name in model_names:
        for dataset_name in _dataset_names.all:
            eval.load_and_display_npy(models_folder, model_name,
                                      dataset_name, img_type='test')


# VERIFY SHAPE OF LOADED IMAGES
def load_and_display_npy(models_folder, model_name, dataset_name, img_type):
    try:
        file = f'{model_name}_{dataset_name}_decoded_{img_type}.npy'
        path = os.path.join(_paths.saves, 'decoded',
                            models_folder, model_name, file)
        data = np.load(path)
        print(f'Loaded data from {dataset_name} decoded ',
              f'by {model_name}. Shape: {data.shape}')
        return data

    except Exception as e:
        print(f"Error loading data from {model_name}: {e}")
        return None


# LOAD HISTORIES
def load_histories_from_directory():
    history_dir = os.path.join(_paths.histories)
    model_histories = {}

    # List all files in the specified directory
    file_names = os.listdir(history_dir)

    for file_name in file_names:
        file_path = os.path.join(history_dir, file_name)

        # Check if the file is a JSON file
        if file_name.endswith('.json'):
            with open(file_path, 'r') as file:
                history = json.load(file)
                # Extract the model name from the file name
                model_name = file_name.split('_history')[0]
                model_histories[model_name] = history

    return model_histories


'''
CALCULATE SSI, PSNR, MAE, MSE
'''


# NORMALIZE BASED ON METRIC
def normalize_images(images, metric_name):
    if metric_name in ['ssi']:  # already in [0,1] after preperation
        return images
    elif metric_name in ['psnr', 'mse', 'mae']:
        return images * 255.0
    else:
        return images


# MEAN SQUARED ERROR
def mse_3d(original, reconstructed, axis=None):
    return np.mean((original - reconstructed)**2, axis=axis)


# MEAN ABSOLUTE ERROR
def mae_3d(original, reconstructed, axis=None):
    return np.mean(np.abs(original - reconstructed), axis=axis)


# CALCULATE MSE AND MAE
def calculate_mse_mae(images_original, images_reconstructed):
    images_original_normalized = normalize_images(
        images_original, 'mse')
    images_reconstructed_normalized = normalize_images(
        images_reconstructed, 'mse')

    mse_scores = []
    mae_scores = []

    for original, reconstructed in zip(images_original_normalized,
                                       images_reconstructed_normalized):
        mse_score = mse_3d(original, reconstructed)
        mae_score = mae_3d(original, reconstructed)

        mse_scores.append(mse_score)
        mae_scores.append(mae_score)

    return np.mean(mse_scores), np.mean(mae_scores)


# CALCULATE STRUCTURAL SIMILARITY INDEX
def calculate_ssi(images_original, images_reconstructed):
    images_original_normalized = normalize_images(
        images_original, 'ssi')
    images_reconstructed_normalized = normalize_images(
        images_reconstructed, 'ssi')

    ssi_scores = []

    for original, reconstructed in zip(images_original_normalized,
                                       images_reconstructed_normalized):
        ssi_score = ssim(original, reconstructed, multichannel=True,
                         channel_axis=-1, data_range=1)
        ssi_scores.append(ssi_score)

    return np.mean(ssi_scores)


# CALCULATE PEAK SIGNAL TO NOISE RATIO
def calculate_psnr(images_original, images_reconstructed):
    images_original_normalized = normalize_images(
        images_original, 'psnr')
    images_reconstructed_normalized = normalize_images(
        images_reconstructed, 'psnr')

    psnr_scores = []

    for original, reconstructed in zip(images_original_normalized,
                                       images_reconstructed_normalized):
        psnr_score = psnr(original, reconstructed, data_range=255.0)
        psnr_scores.append(psnr_score)

    return np.mean(psnr_scores)


# CALCULATE ALL METRICS AND APPEND TO YAML
def calculate_and_append_metrics(all_datasets, model_name):
    for dataset_name, (original_images,
                       decoded_images) in all_datasets.items():

        ssi_score = calculate_ssi(original_images, decoded_images)
        psnr_score = calculate_psnr(original_images, decoded_images)
        mse_score, mae_score = calculate_mse_mae(original_images,
                                                 decoded_images)

        metric_scores = {'ssi': ssi_score, 'psnr': psnr_score,
                         'mse': mse_score, 'mae': mae_score}

        append_entry_to_metric_yaml(model_name, dataset_name, metric_scores)


# EVALUATION FOR MODELS IN FOLDER
def evaluation_for_models_in_folder(models_folder, model_names,
                                    amount=500,
                                    img_type='test',
                                    datasets=_dataset_names.to_evaluate):
    if datasets is None:
        print('Define dataset to decode for evaluation')
        return
   
    for model_name in model_names:
        ae, _, _ = model_train.loading_models_and_history(models_folder,
                                                          model_name)

        all_datasets = process_images_and_collect(datasets,
                                                  amount,
                                                  img_type,
                                                  models_folder,
                                                  model_name,
                                                  ae)
        calculate_and_append_metrics(all_datasets, model_name)


'''
VISUALIZE EVALUATION
'''


# CREATE METRIC COMPARISON TABLE
def create_comparison_table(yaml_path, metric_name):
    with open(yaml_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    dfs = []
    for model_name, model_data in data['models'].items():
        model_scores = {f'{dataset}_{metric_name}':
                        float(score[metric_name])
                        for dataset, score
                        in model_data.items()}

        df_model = pd.DataFrame(model_scores, index=[model_name])
        dfs.append(df_model)

    result_df = pd.concat(dfs)
    result_df['mean_score'] = result_df.mean(axis=1)
    return result_df


# COMPARE MODEL HISTORIES
def compare_model_histories(history_list, metric_name='loss',
                            selected_epochs=None):
    df = pd.DataFrame()
    selected_epochs = selected_epochs or range(0, 500, 25)

    for model_name, history in history_list.items():
        metric_values = history.get(metric_name, [])
        selected_values = [metric_values[epoch - 1]
                           if epoch <= len(metric_values)
                           else None for epoch in selected_epochs]

        model_df = pd.DataFrame({
            'Epoch': selected_epochs,
            f'{model_name}_{metric_name}': selected_values
        })

        df = pd.concat([df, model_df.set_index('Epoch')], axis=1)
    df = df.transpose()

    return df


# VISUALIZE PLOT LOSS
def plot_loss(history_dir, save):
    file_names = os.listdir(history_dir)

    for file_name in file_names:
        if file_name.endswith('_history.json'):
            model_name = file_name.split('_history')[0]
            history_path = os.path.join(history_dir, file_name)

            with open(history_path, 'r') as file:
                history = json.load(file)

            plt.plot(range(1, len(history['loss']) + 1),
                     history['loss'], label='Training Loss')
            plt.plot(range(1, len(history['val_loss']) + 1),
                     history['val_loss'], label='Validation Loss')

            plt.title(f'Train and Val Loss of {model_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            if save:
                model_output_saves = os.path.join(
                    _paths.output, 'plot_loss')

                os.makedirs(model_output_saves, exist_ok=True)

                img_to_save = os.path.join(
                    model_output_saves, f'{model_name}_loss.png')

                plt.savefig(img_to_save)

            plt.show()
