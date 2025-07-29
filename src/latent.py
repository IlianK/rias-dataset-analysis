import gc
import math
import os
from matplotlib import pyplot as plt
import yaml
from utils import _paths, _dataset_names
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from keras.models import Model
import model_train
from IPython.display import display
import data


'''
ENCODING TO LATENT SPACE
'''


# SHOW ACTIVATION OF SPECIFIC LAYER
def activation_for_img(activation, layer_name):
    _, _, _, channels = activation.shape
    
    cols = 8
    rows = np.ceil(channels / cols).astype(int)  
    
    plt.figure(figsize=(16, rows * 2))
    for i in range(channels):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(activation[0, :, :, i], cmap='viridis')
        plt.title(f"Ch {i}")
        plt.axis('off')

    plt.suptitle(f'Activation of {layer_name}', fontsize=20)
    plt.tight_layout()
    plt.show()


# GET ACTIVATION OF SPECIFIC LAYER
def get_layer_activation(encoder_model, input_image, layer_name):
    activation_model = Model(
        inputs=encoder_model.input,
        outputs=encoder_model.get_layer(layer_name).output)
    input_image_batch = np.expand_dims(input_image, axis=0)
    activations = activation_model.predict(input_image_batch)
    return activations


def get_encoder_activations(encoder_model, input_image):
    activation_model = Model(
        inputs=encoder_model.input,
        outputs=[layer.output for layer in encoder_model.layers])
    input_image_batch = np.expand_dims(input_image, axis=0)
    activations = activation_model.predict(input_image_batch)
    return activations


# ENCODE IMAGES WITH ENCODER MODEL
def encode(encoder_model, images):
    encoded_batch = encoder_model.predict(images)
    print(encoded_batch.shape)
    return encoded_batch


# ENCODE ENTIRE DATASETS
def encode_dataset(dataset_name, encoder_model):
    def encode_files_in_path(path):
        file_paths = [os.path.join(path, file) for file in os.listdir(path)]
        batch_data = [np.load(file_path) for file_path in file_paths]
        return encoder_model.predict(np.array(batch_data))

    base_path = os.path.join(_paths.datasets_prepared_whole, dataset_name)

    train_encoded = encode_files_in_path(os.path.join(base_path, 'train'))
    val_encoded = encode_files_in_path(os.path.join(base_path, 'val'))
    test_encoded = encode_files_in_path(os.path.join(base_path, 'test'))

    return np.concatenate([train_encoded, val_encoded, test_encoded], axis=0)


# ENCODE AND SAVE DATASETS
def encode_and_save_datasets(dataset_names, encoder, model_folder, model_name):
    if dataset_names is None:
        print('Define dataset to encode')
        return

    for dataset_name in dataset_names:
        encoded_dataset = encode_dataset(dataset_name, encoder)
        encoded_path = os.path.join(_paths.saves, 'encoded', model_folder,
                                    f'{model_name}_{dataset_name}_encoded.npy')
        os.makedirs(os.path.dirname(encoded_path), exist_ok=True)
        np.save(encoded_path, encoded_dataset)
        print(f'Dataset {dataset_name} was encoded ',
              f'with shape: {encoded_dataset.shape}')
        gc.collect()


# ENCODE WITH ALL
def encode_datasets_with_all_models():
    model_info_yaml_path = os.path.join(_paths.yaml_files.model_info)
    with open(model_info_yaml_path, 'r') as yaml_file:
        model_info = yaml.safe_load(yaml_file)
        
    for model_dir, model_names in model_info.items():
        for model_name in model_names:
            _, encoder, _ = model_train.loading_models_and_history(model_dir, model_name)
            encode_and_save_datasets(_dataset_names.to_encode,
                                     encoder, model_dir, model_name)


# LOAD ENCODED DATASETS
def load_encoded_datasets(dataset_names, model_name, models_folder):
    datasets = {}

    for dataset_name in dataset_names:
        encoded_path = os.path.join(_paths.encoded,
                                    models_folder, model_name,
                                    f'{model_name}_{dataset_name}_encoded.npy')

        encoded_dataset = np.load(encoded_path)
        datasets[dataset_name] = encoded_dataset

        print(f'Dataset {dataset_name} loaded with shape: '
              f'{encoded_dataset.shape}')

    return datasets


'''
DIMENSION REDUCTION
'''


# CALUCLATE DIMENSION REDUCTION
def calculate_dimension_reduction(original, latent_space):
    original_dimension = np.prod(original)
    latent_dimension = np.prod(latent_space)

    reduction = (original_dimension - latent_dimension) / original_dimension
    reduction_percentage = reduction * 100
    print(f'Original: {original}, Latent_Space: {latent_space}, '
          f'Dimensionality Reduction: {reduction_percentage:.2f}%')


# CALCULATE SIMPLE RESIZE DIMENSIONS
def calculate_simple_resize_dimensions(original_shape, target_latent_shape):
    target_latent_size = np.prod(target_latent_shape)

    current_dimensions = original_shape[:2]
    current_size = np.prod(current_dimensions)

    while current_size >= target_latent_size:
        current_dimensions = (current_dimensions[0] // 2,
                              current_dimensions[1] // 2)

        current_size = np.prod(current_dimensions)

    resized_dimensions = (current_dimensions[0],
                          current_dimensions[1],
                          original_shape[-1])

    print(f'Original: {original_shape}, Latent_Space: {target_latent_shape}, '
          f'Resized dimensions: {resized_dimensions} '
          f'= {np.prod(resized_dimensions)}')

    return resized_dimensions


'''
LATENT SPACE VISUALIZATION
'''


# SHOW LATENT DIM FOR IMAGES N TO M
def latent_dim_for_img_range(dataset_name, encoded_datasets,
                             num_images, dim):

    title = f'Channel {dim} for {num_images} encoded images of dataset {dataset_name}'

    encoded_imgs = encoded_datasets[dataset_name][:num_images]
    num_rows = math.ceil(num_images / 8)

    plt.figure(figsize=(16, num_rows * 2))

    for i, idx in enumerate(range(num_images)):
        plt.subplot(num_rows, 8, i + 1)
        plt.imshow(encoded_imgs[i, :, :, dim])  # Display images in grayscale
        plt.title(f"Img {idx}")
        plt.axis('off')

    plt.suptitle(title, fontsize=20)
    plt.show()


'''
FLATTENED DATA (DATASET AND AGGREGATED FEATURE VECTOR)
'''


# GET FLATTENED DATA OF DATASET
def get_all_flattened_ds(encoded_data_dict, print_shape=False):
    all_flattened_ds = {}

    for dataset_name in encoded_data_dict:
        encoded_data = encoded_data_dict[dataset_name]
        flat_encoded_data = encoded_data.reshape(encoded_data.shape[0], -1)

        if print_shape:
            print(f'Encoded dataset shape: {encoded_data.shape}')
            print(f'Flattened encoded dataset shape: '
                  f'{flat_encoded_data.shape}')

        all_flattened_ds[dataset_name] = flat_encoded_data

    return all_flattened_ds


# GET FLATTENED DATA OF FEATURE VECTOR
def get_all_flattened_fv(encoded_data_dict, metric, print_shape=False):
    all_flattened_fv = {}

    for dataset_name in encoded_data_dict:
        encoded_data = encoded_data_dict[dataset_name][metric]
        flat_encoded_data = encoded_data.reshape(-1, encoded_data.shape[2])

        if print_shape:
            print(f'Encoded {metric} feature vector shape: ',
                  f'{encoded_data.shape}')
            print(f'Flattened encoded {metric} feature vector shape: ',
                  f'{flat_encoded_data.shape}')

        all_flattened_fv[dataset_name] = flat_encoded_data

    return all_flattened_fv


'''
CREATE NEW ENCODER MODEL
'''


def print_layers(encoder):
    layer_names = [layer.name for layer in encoder.layers]
    print(layer_names)


def cut_encoder_to_new_model(output_layer, model_dir, model_name, save=False,
                             show_summary=False):
    new_encoder_name = f'{model_name}_encoder'

    _, old_encoder, _ = model_train.loading_models_and_history(
        model_dir, model_name)

    print_layers(old_encoder)

    encoder_input = old_encoder.input
    desired_output_layer = old_encoder.get_layer(output_layer).output
    new_encoder = Model(inputs=encoder_input, outputs=desired_output_layer,
                        name=new_encoder_name)

    model_path = os.path.join(_paths.models, 'encoder_models')
    if save:
        new_encoder.save(f'{model_path}/{new_encoder_name}.h5',
                         save_format='h5')

    if show_summary:
        new_encoder.summary()

    return new_encoder


'''
VISUALIZE AGGREGATED DATA
'''


# DISPLAY METRIC PER FEATURE
def display_metric(metric, all_features):
    flattened_metric = {}
    for key, subdict in all_features.items():
        flattened_metric[key] = subdict[metric]

    metric_df = pd.DataFrame(flattened_metric)
    pd.options.display.float_format = '{:.4f}'.format
    display(metric_df)


def display_statistics(dataset_statistics):
    dataset_statistics_no_dtype = {key: {subkey: float(value) for subkey,
                                         value in subdict.items()} for key,
                                   subdict in dataset_statistics.items()}

    statistics_df = pd.DataFrame(dataset_statistics_no_dtype)
    pd.options.display.float_format = '{:.4f}'.format
    display(statistics_df.T)


# SHOW LATENT SPACE (ALL DIMS) FOR IMG
def latent_space_for_img(dataset_name, encoded_datasets,
                         model_name,
                         metric, save=False):

    encoded_img = encoded_datasets[dataset_name][metric]
    _, _, channels = encoded_img.shape

    if isinstance(metric, int):
        title = f'{channels} feature maps of image num {metric} '
        title = title + f'of dataset {dataset_name}'
    else:
        title = f'{metric} feature maps of dataset {dataset_name}'

    plt.figure(figsize=(16, 16))

    for i in range(channels):
        plt.subplot(8, 8, i + 1)
        plt.imshow(encoded_img[:, :, i], cmap='viridis')
        plt.title(f"Ch {i}", fontsize=10)
        plt.axis('off')

    plt.suptitle(title, fontsize=20, y=0.92)

    if save:
        save_path = os.path.join(_paths.output, 'latent_space',
                                 model_name, metric)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f'{dataset_name}_{metric}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)

    plt.show()


# PLOT ACTIVATION
def plot_activation_histogram(dataset_name, aggregate_type, all_vectors):
    aggregated_values = all_vectors[dataset_name][aggregate_type]
    flattened_values = aggregated_values.flatten()

    plt.hist(flattened_values, bins=50)
    plt.title(f'Histogram of {aggregate_type} activations for {dataset_name}')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')

    plt.show()


'''
VISUALIZE DIMENSIONS COMPARISON
'''


# COMPARE DIMENSIONS X Y OF ENCODED DATA
def compare_dims_xy(encoded_data, x, y):
    title = f'Visualization of Latent Space (Dimensions {x} and {y}); '
    title += f'shape: {encoded_data.shape}'

    fig = px.scatter(x=encoded_data[:, x], y=encoded_data[:, y], title=title)
    fig.update_layout(xaxis_title=f'Dim {x}', yaxis_title=f'Dim {y}')
    fig.show()


# COMPARE ONE DIMENSION AGAINST RANGE OF OTHER DIMENSIONS
def compare_dimx_against_dims(encoded_data, dim_x, dims):
    num_plots = len(dims)

    fig = make_subplots(rows=(num_plots - 1) // 4 + 1,
                        cols=4,
                        subplot_titles=[f'Dim {dim_x} vs. Dim {comp_dim}'
                                        for comp_dim in dims])

    for idx, comp_dim in enumerate(dims):
        row = idx // 4 + 1
        col = idx % 4 + 1

        fig.add_trace(go.Scatter(x=encoded_data[:, dim_x],
                                 y=encoded_data[:, comp_dim],
                                 mode='markers',
                                 marker=dict(size=4, color='#d53a26'),
                                 showlegend=False),
                      row=row, col=col)

        fig.update_xaxes(title_text=f'Dimension {dim_x}', row=row, col=col)
        fig.update_yaxes(title_text=f'Dimension {comp_dim}', row=row, col=col)

    fig.update_layout(height=400 * ((num_plots - 1) // 4 + 1),
                      title_text=f'Dim {dim_x} vs. Various Dimensions')
    fig.show()


# VISUALIZE ENCODED DATA IN 3D
def visualize_3d(encoded_data, x, y, z):
    x = encoded_data[:, x]
    y = encoded_data[:, y]
    z = encoded_data[:, z]

    scatter_3d = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                              marker=dict(size=3, color=z,
                                          colorscale='Viridis',))

    layout = go.Layout(scene=dict(aspectmode="cube"))
    fig = go.Figure(data=[scatter_3d], layout=layout)

    fig.update_layout(scene=dict(xaxis_title='PC1',
                                 yaxis_title='PC2',
                                 zaxis_title='PC3'))
    fig.show()


'''
RESIZE COMPARISON
'''


# PLOT IMAGES
def plot_images(images, title_prefix=''):
    images = images.astype(float) / images.max()
    for i, image in enumerate(images):
        image = (image * 255.).astype(np.uint8)

        rgb_image = image[:, :, :3]
        mask = image[:, :, 3]

        plt.subplot(2, len(images), i + 1)
        plt.imshow(rgb_image)
        plt.title(f'{title_prefix} RGB {i + 1}')
        plt.axis('off')

        plt.subplot(2, len(images), i + len(images) + 1)
        plt.imshow(mask, cmap='gray')
        plt.title(f'{title_prefix} Mask {i + 1}')
        plt.axis('off')

    plt.show()


# PREPROCESS IMAGE
def preprocess_image(input_image):
    rgb_img = input_image[..., :3]
    mask_channel = input_image[..., 3]
    data.display_img_mask(rgb_img, mask_channel)


# GET SEPARATE CHANNELS  
def get_separate_channels(input_image):
    rgb_img = input_image[..., :3]
    mask_channel = input_image[..., 3]
    return rgb_img, mask_channel


# DISPLAY IMAGES
def display_images(rgb_img, mask_channel, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title('RGB Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask_channel, cmap='gray')
    plt.title('Mask Channel')
    plt.axis('off')

    plt.suptitle(title)
    plt.show()
    
    
# PREPROCESS AND
def preprocess_and_display_images(input_image, decoded_image, title_prefix=''):
    input_rgb_img, input_mask_channel = get_separate_channels(input_image)
    decoded_rgb_img, decoded_mask_channel = get_separate_channels(
        decoded_image)

    display_images(input_rgb_img,
                   input_mask_channel,
                   f'{title_prefix} Input Image')
    display_images(decoded_rgb_img,
                   decoded_mask_channel,
                   f'{title_prefix} Decoded Image')
