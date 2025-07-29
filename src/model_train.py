import datetime
import random
from utils import _paths, _params, separator
from matplotlib import pyplot as plt
import os
import numpy as np
import json
import yaml
import gc
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, model_from_json
from keras.layers import (
    Input, Conv2D, Conv2DTranspose, BatchNormalization,
    Dense, Activation, LeakyReLU, Add, Flatten, Reshape
)


'''
PREPARE DATA FOR TRAINING
'''


# GET SUBSET OF TRAIN AND VAL FOR TRAINING
def get_train_val_subset(dataset_name, n_to_train):
    subset_paths_dict = {}
    base_path = os.path.join(_paths.datasets_prepared, dataset_name)
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'val')

    train_files = os.listdir(train_path)
    val_files = os.listdir(val_path)

    num_train_images = min(len(train_files), n_to_train)
    num_val_images = int(num_train_images * _params.val_size)

    train_subset = train_files[:num_train_images]
    val_subset = val_files[:num_val_images]

    subset_paths_dict['train'] = [os.path.join(_paths.datasets_prepared,
                                               dataset_name, 'train', file)
                                  for file in train_subset]

    subset_paths_dict['val'] = [os.path.join(_paths.datasets_prepared,
                                             dataset_name, 'val', file)
                                for file in val_subset]

    enough_train_imgs = len(subset_paths_dict['train']) < _params.batch_size
    enough_val_imgs = len(subset_paths_dict['val']) < _params.batch_size

    if enough_train_imgs or enough_val_imgs:
        raise ValueError("Number of train and val images must be at least "
                         f"{_params.batch_size} (batch size).")

    return {dataset_name: subset_paths_dict}


# CREATE DATA GENERATOR FOR AUTOENCODER
def create_autoencoder_generator(file_paths, data_type):
    rng = np.random.default_rng(seed=42)
    rng.shuffle(file_paths)
    num_batches = len(file_paths) // _params.batch_size

    for i in range(num_batches):
        start_index = i * _params.batch_size
        end_index = (i + 1) * _params.batch_size
        batch_paths = file_paths[start_index:end_index]
        batch_data = []

        for file_path in batch_paths:
            data = np.load(file_path)
            batch_data.append(data)

        batch_data = np.array(batch_data)

        # print("Loaded {} batch: {}, Shape: {}".format(
        #    data_type,
        #    [os.path.basename(path) for path in batch_paths],
        #    batch_data.shape
        # ))

        yield batch_data, batch_data


# CUSTOM CALLBACKS
def get_callbacks(model_name):
    callbacks = []

    if _params.use_checkpoint:
        model_path = os.path.join(_paths.models, model_name)
        checkpoints_dir = os.path.join(model_path, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        filepath = os.path.join(checkpoints_dir, f'{model_name}_checkpoint.h5')
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        callbacks.append(checkpoint)

    if _params.use_early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=3,
                                       restore_best_weights=True)
        callbacks.append(early_stopping)

    if _params.use_tensorboard_logs:
        log_folder_name = 'logs_' + model_name
        log_folder_name += datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        log_folder = os.path.join(_paths.logs, log_folder_name)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_folder,
                                                     histogram_freq=1)
        callbacks.append(tensorboard)

    return callbacks


'''
MODEL DEFINITION
'''


# GENERAL MODEL FUNCTIONS
def conv_block(x, filters, strides=(1, 1), name_prefix=""):
    x = Conv2D(filters, (3, 3), strides=strides,
               padding='same', name=f"{name_prefix}_conv")(x)
    x = BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = LeakyReLU(name=f"{name_prefix}_relu")(x)
    return x


def deconv_block(x, filters, strides=(1, 1), name_prefix=""):
    x = Conv2DTranspose(filters, (3, 3), strides=strides,
                        padding='same', name=f"{name_prefix}_deconv")(x)
    x = BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = LeakyReLU(name=f"{name_prefix}_relu")(x)
    return x


def deep_conv_block(x, filters, name_prefix=""):
    x = conv_block(x, filters, name_prefix=f"{name_prefix}_conv1")
    x = conv_block(x, filters, name_prefix=f"{name_prefix}_conv2")
    x = conv_block(x, filters, name_prefix=f"{name_prefix}_conv3")
    return x


def dense_bn_relu(x, units, name_prefix=""):
    x = Dense(units=units, name=f"{name_prefix}_dense")(x)
    x = BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = Activation('relu', name=f"{name_prefix}_relu")(x)
    return x


def residual_block(x, filters, name_prefix=""):
    residual = x
    x = deep_conv_block(x, filters, name_prefix=f"{name_prefix}")
    x = Add(name=f"{name_prefix}_add")([x, residual])
    x = Activation('relu', name=f"{name_prefix}_relu")(x)
    return x


def res_or_deep(x, filters, i, name, ae_type='r_cae'):
    if ae_type == 'r_cae':
        x = residual_block(x, filters[i],
                           name_prefix=f"deep_res_{name}{i}")
    else:
        x = deep_conv_block(x, filters[i],
                            name_prefix=f"deep_conv_{name}{i}")
    return x


# BUILD ENCODER
def build_encoder(input_layer, filters, ae_type):
    x = input_layer

    if filters:
        for i in range(len(filters)):
            if i == 0:
                x = conv_block(x, filters[0], strides=(1, 1),
                               name_prefix="block")
                x = res_or_deep(x, filters, i, 'down', ae_type)

            if i != (len(filters) - 1):
                x = conv_block(x, filters[i + 1], strides=(2, 2),
                               name_prefix=f"conv_block_down{i}")
                x = res_or_deep(x, filters, i+1, 'down', ae_type)
                
            if i == (len(filters) - 1):
                x = conv_block(x, filters[i], strides=(2, 2),
                               name_prefix=f"conv_block_down{i}")               
                x = res_or_deep(x, filters, i, 'latent_space', ae_type)

    return x


# BUILD LATENT SPACE
def build_latent_space(x, dense_units):
    if dense_units:
        target_shape = x.shape
        flattened_target = np.prod(target_shape[-3:])
        x = Flatten(name="flatten_target_for_dense")(x)

        if len(dense_units) > 1:
            for units in dense_units[-1:]:
                x = dense_bn_relu(x, units, name_prefix=f"dense_down_{units}")

        x = dense_bn_relu(x, dense_units[-1],
                          name_prefix=f"dense_latent_space_{dense_units[-1]}")
        encoded = x

        if len(dense_units) > 1:
            for units in reversed(dense_units[-1:]):
                x = dense_bn_relu(x, units, name_prefix=f"dense_up_{units}")

        x = Dense(units=flattened_target, activation='sigmoid',
                  name="dense_up_to_flattened_target")(x)
        x = Reshape(target_shape[-3:], name="reshape_flattened_target")(x)
    else:
        encoded = x

    return x, encoded


# BUILD DECODER
def build_decoder(x, filters, ae_type):
    output_layer = x
    if filters:
        for i in range(len(filters) - 1, -1, -1):
            if i != 0:
                x = deconv_block(x, filters[i], strides=(2, 2),
                                 name_prefix=f"conv_block_up{i}")
                x = res_or_deep(x, filters, i, 'up', ae_type)
            if i == 0:
                x = deconv_block(x, filters[0], strides=(2, 2),
                                 name_prefix=f"conv_block_up{i}")
                x = res_or_deep(x, filters, 0, 'up', ae_type)

        output_layer = Conv2D(4, (3, 3), padding="same",
                              activation="sigmoid", name="decoded")(x)

    return output_layer


# BUILD AUTOENCODER
def build_ae(_input_shape, filters, model_name, ae_type, dense_units=None):
    # Encoder
    input_layer = Input(_input_shape, name="input_layer")
    encoder_output = build_encoder(input_layer, filters, ae_type)

    # Latent Space
    latent_output, encoded = build_latent_space(encoder_output,
                                                dense_units)

    # Decoder
    decoder_output = build_decoder(latent_output, filters, ae_type)

    # Models
    encoder = Model(inputs=input_layer, outputs=encoded,
                    name=f'{model_name}_encoder')
    decoder = Model(inputs=encoded, outputs=decoder_output,
                    name=f'{model_name}_decoder')

    autoencoder_output = decoder(encoder(input_layer))
    autoencoder = Model(inputs=input_layer, outputs=autoencoder_output,
                        name=f'{model_name}_autoencoder')

    # Compile
    autoencoder.compile(optimizer=_params.optimizer,
                        loss=_params.loss,
                        metrics=_params.metrics)

    return autoencoder, encoder, decoder


'''
MODEL TRAINING
'''


# COMBINE HISTORIES TO DICTIONARY
def combine_histories_to_dict(histories):
    combined_history = {}
    for key in histories[0].keys():
        combined_history[key] = []
        for history in histories:
            if isinstance(history[key], list):
                combined_history[key].extend(history[key])
            else:
                combined_history[key].append(history[key])

    return combined_history


# GET MODEL PATHS
def get_model_paths(model_dir, model_name, model_type):
    model_type_dir = os.path.join(model_dir, model_type)
    architecture_path = os.path.join(
        model_type_dir, f"{model_name}_{model_type}_architecture.json"
    )
    weights_path = os.path.join(
        model_type_dir, f"{model_name}_{model_type}_weights.h5"
    )

    return architecture_path, weights_path


# GET HISTORY PATH
def get_history_path(model_dir, model_name):
    return os.path.join(model_dir, f"{model_name}_history.json")


# SAVE TRAINED MODEL TO JSON
def save_model_to_json(model, architecture_path, weights_path):
    model_json = model.to_json()
    with open(architecture_path, "w") as json_file:
        json_file.write(model_json)

    model.save_weights(weights_path)


# SAVE MODELS AND HISTORY
def saving_models_and_history(autoencoder_model, encoder_model,
                              model_name, all_histories):
    combined_history = combine_histories_to_dict(all_histories)
    model_dir = os.path.join(_paths.models, model_name)

    for model_type in ['autoencoder', 'encoder']:
        model_type_dir = os.path.join(model_dir, model_type)
        os.makedirs(model_type_dir, exist_ok=True)

        architecture_path, weights_path = get_model_paths(model_dir,
                                                          model_name,
                                                          model_type)

        save_model_to_json(locals()[f"{model_type}_model"],
                           architecture_path, weights_path)

    history_path = get_history_path(model_dir, model_name)
    with open(history_path, "w") as history_file:
        json.dump(combined_history, history_file)


# LOAD TRAINED MODEL
def load_model_from_json(model_type, model_dir, model_name):
    architecture_path, weights_path = get_model_paths(model_dir,
                                                      model_name,
                                                      model_type)

    with open(architecture_path, "r") as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model


# LOAD HISTORY
def load_history(history_path):
    with open(history_path, "r") as history_file:
        return json.load(history_file)


# LOAD MODELS AND HISTORY
def loading_models_and_history(models_folder, model_name):
    model_dir = os.path.join(_paths.models, models_folder, model_name)

    autoencoder = load_model_from_json('autoencoder', model_dir, model_name)
    encoder = load_model_from_json('encoder', model_dir, model_name)

    history_path = get_history_path(model_dir, model_name)
    history_exists = os.path.exists(history_path)
    history = load_history(history_path) if history_exists else None

    return autoencoder, encoder, history


'''
MODEL TRAINING HELPER
'''


# CONVERT COMBINED HISTORY BACK TO LIST FOR TRAINING
def convert_histories_to_list(combined_history):
    keys = list(combined_history.keys())
    num_epochs = len(combined_history[keys[0]])

    all_histories_list = []

    for epoch in range(num_epochs):
        epoch_history = {}
        for key in keys:
            epoch_history[key] = combined_history[key][epoch]
        all_histories_list.append(epoch_history)

    return all_histories_list


# DATASET INFO
def print_dataset_info(train_data, val_data):
    print(separator)

    train_size, val_size = len(train_data), len(val_data)
    train_batches = train_size // _params.batch_size
    val_batches = val_size // _params.batch_size

    print(f'Train images: {train_size} - Batches: {train_batches}')
    print(f'Val images: {val_size} - Batches: {val_batches}')

    print(separator)


# TRAIN AUTOENCODER
def train_autoencoder(autoencoder, _train_params, epoch, dataset_name):
    n_imgs_to_train = (
        _train_params.real_amount
        if dataset_name == 'adidas_real'
        else _train_params.imgs_to_train_on
    )

    subset_paths_dict = get_train_val_subset(dataset_name, n_imgs_to_train)
    train_data = subset_paths_dict[dataset_name]['train']
    val_data = subset_paths_dict[dataset_name]['val']

    callbacks = get_callbacks(_train_params.model_name)
    print_dataset_info(train_data, val_data)

    print(f'Epoch: {epoch + 1}/{_train_params.epochs_per_dataset}')

    train_generator = create_autoencoder_generator(train_data, 'train')
    val_generator = create_autoencoder_generator(val_data, 'val')

    history = autoencoder.fit(
        train_generator,
        steps_per_epoch=len(train_data) // _params.batch_size,
        epochs=1,
        verbose=2,
        shuffle=_params.shuffle,
        validation_data=val_generator,
        validation_steps=len(val_data) // _params.batch_size,
        callbacks=callbacks,
    )

    train_steps = len(train_data) // _params.batch_size
    val_steps = len(val_data) // _params.batch_size
    print(f'Training steps: {train_steps} - Validation steps: {val_steps}')

    return history


# SAVE AUTOENCODER
def save_autoencoder(autoencoder, encoder, _train_params, histories):
    print('Saving models and history of:', _train_params.model_name)
    saving_models_and_history(autoencoder, encoder,
                              _train_params.model_name,
                              histories)
    print(separator)


# TRAIN AND SAVE AUTOENCODER
def train_and_save_autoencoder(autoencoder, encoder,
                               _train_params,
                               history=None):

    histories = convert_histories_to_list(history) if history else []

    for dataset_name in _train_params.dataset_names:
        for epoch in range(_train_params.epochs_per_dataset):
            history = train_autoencoder(autoencoder, _train_params,
                                        epoch, dataset_name)
            histories.append(history.history)
            save_autoencoder(autoencoder, encoder, _train_params, histories)
            gc.collect()

# CLASS FOR TRAINING PARAMETERS
class TrainParams:
    def __init__(self, model_name, filters, real_amount,
                 dataset_names, imgs_to_train_on, epochs_per_dataset,
                 autoencoder_type, dense_units):

        self.model_name = model_name
        self.filters = filters
        self.real_amount = real_amount
        self.dataset_names = dataset_names
        self.imgs_to_train_on = imgs_to_train_on
        self.epochs_per_dataset = epochs_per_dataset
        self.autoencoder_type = autoencoder_type
        self.dense_units = dense_units


# TRAIN AUTOENCODER WITH PARAMETERS
def new_ae_training_with_params():
    with open(_paths.yaml_files.training_param, 'r') as file:
        parameters = yaml.safe_load(file)

    for _, params in parameters.items():
        _train_params = TrainParams(
            model_name=params['model_name'],
            filters=params['filters'],
            real_amount=params['real_amount'],
            dataset_names=params['dataset_names'],
            imgs_to_train_on=params['imgs_to_train_on'],
            epochs_per_dataset=params['epochs_per_dataset'],
            autoencoder_type=params['autoencoder_type'],
            dense_units=params['dense_units']
            )

        autoencoder, encoder, decoder = build_ae(
            _params.input_shape,
            _train_params.filters,
            _train_params.model_name,
            _train_params.autoencoder_type,
            _train_params.dense_units)

        print(f"Training: {_train_params.model_name}")
        print(autoencoder.summary())
        print(encoder.summary())
        print(decoder.summary())

        train_and_save_autoencoder(autoencoder, encoder, _train_params)
        gc.collect()


# CONTINUE FAILED TRAINING
def continue_failed_training():

    with open(_paths.yaml_files.general_params, 'r') as file:
        general_params = yaml.safe_load(file)

    models_folder = os.path.join(general_params['models_folder'],
                                 general_params['models_subfolder'])

    model_name = general_params['model_name']
    failed_at_epoch = general_params['failed_at_epoch']
    missing_datasets = general_params['missing_datasets']

    with open(_paths.yaml_files.training_param, 'r') as file:
        params = yaml.safe_load(file)

    matching_entry = None
    for key, value in params.items():
        if 'model_name' in value and value['model_name'] == model_name:
            matching_entry = key
            break

    if matching_entry is None:
        raise ValueError(f"Model name '{model_name}' not found in YAML file.")

    model_params = params[matching_entry]
    epoch_to_continue = model_params['epochs_per_dataset'] - failed_at_epoch
    model_params['epochs_per_dataset'] = epoch_to_continue
    model_params['dataset_names'] = missing_datasets

    _train_params = TrainParams(
            model_name=model_params['model_name'],
            filters=model_params['filters'],
            real_amount=model_params['real_amount'],
            dataset_names=model_params['dataset_names'],
            imgs_to_train_on=model_params['imgs_to_train_on'],
            epochs_per_dataset=model_params['epochs_per_dataset'],
            autoencoder_type=model_params['autoencoder_type'],
            dense_units=model_params['dense_units']
            )

    autoencoder, encoder, history = loading_models_and_history(
        models_folder,
        model_name)

    autoencoder.compile(optimizer=_params.optimizer,
                        loss=_params.loss,
                        metrics=_params.metrics)

    print(f'Continue training at epoch {failed_at_epoch} '
          f'for model: {model_name}')

    train_and_save_autoencoder(autoencoder, encoder, _train_params, history)


'''
AE TRAINING RESULTS VISUALIZATION
'''


# LOAD EXAMPLE BATCH FOR VISUALIZATION
def load_example_batch(dataset_name, img_type, use_random=False):
    batch = []

    if img_type in ['train', 'val', 'test']:
        base_path = _paths.datasets_prepared_whole
        img_type_dir = os.path.join(base_path, dataset_name, img_type)

    all_files = os.listdir(img_type_dir)

    if use_random:
        random_index = random.randint(0, len(all_files) - 1)
        file_name = all_files[random_index]
    elif all_files:
        file_name = all_files[0]
    else:
        file_name = None

    if file_name:
        file_path = os.path.join(img_type_dir, file_name)
        print(f'Loaded img: {file_path}')
        batch = np.load(file_path)[np.newaxis, ...]

    return batch


# DISPLAY RECONSTRUCTED IMAGE AND MASK
def display_reconstructed_and_original_images(data, autoencoder, dataset_name,
                                              save, model_name):
    decoded = autoencoder.predict(np.expand_dims(data, axis=0))
    plt.figure(figsize=(15, 10))
    
    original_rgb_img, original_mask_channel = data[..., :3], data[..., 3]
    reconstructed_rgb_img, reconstructed_mask_channel = decoded[..., :3], decoded[..., 3]

    for i, (image, title, cmap) in enumerate([
        (original_rgb_img * 255, "Original RGB", None),
        (original_mask_channel * 255, "Original Mask", 'gray'),
        (reconstructed_rgb_img[0] * 255, "Decoded RGB", None),
        (reconstructed_mask_channel[0] * 255, "Decoded Mask", 'gray')
    ]):
        plt.subplot(2, 2, i + 1)
        plt.imshow(image.astype(np.uint8), cmap=cmap)
        plt.title(title)
        plt.axis('off')

    if save:
        path = os.path.join(_paths.output, 'reconstructions', model_name)
        os.makedirs(path, exist_ok=True)
        img = os.path.join(path, f'{model_name}_{dataset_name}.png')
        plt.savefig(img)

    plt.show()


# DISPLAY SUBPLOTS
def display_subplot(image, title, pos, cmap):
    plt.subplot(pos)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')


# DISPLAY RECONSTRUCTED AND ORIGINAL IMAGES FOR DATASETS
def display_test_imgs(samples, dataset_names, autoencoder, save, model_name):
    for dataset_name in dataset_names:
        print(f'Dataset: {dataset_name}')
        for data in samples[dataset_name]:
            display_reconstructed_and_original_images(data, autoencoder,
                                                      dataset_name, save,
                                                      model_name)


# LOAD ALL MODELS TO RECONSTRUCT SAMPLES
def load_all_models_to_reconstruct(samples, model_info,
                                   dataset_names, save=False):
    for model_dir, models in model_info.items():
        for model_name in models:
            print(f'Reconstructions for model {model_name}')
            autoencoder, _, _ = loading_models_and_history(model_dir,
                                                           model_name)
            display_test_imgs(samples, dataset_names,
                              autoencoder, save, model_name)
            