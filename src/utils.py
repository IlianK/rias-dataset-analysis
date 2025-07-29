# IMPORTS
import os
from sysconfig import get_python_version
import yaml
import tensorflow as tf


# GLOBAL PARAMS
separator = '**********************************************'


"""
PATHS, PARAMS, GPU VERIFICATION
"""


# DATASET NAMES
class DatasetNames:
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        (
            self.base,
            self.test,
            self.all,
            self.to_evaluate,
            self.to_encode,
            self.to_decode,
            self.object_detection_datasets,
            self.bosch_datasets
        ) = self._load_dataset_names()

    def _load_dataset_names(self):
        with open(self.yaml_path, 'r') as file:
            dataset_config = yaml.safe_load(file)

        base_datasets = dataset_config.get('base_datasets', [])
        test_datasets = dataset_config.get('test_datasets', [])  
        variant_datasets = dataset_config.get('variant_datasets', [])
        all_datasets = base_datasets + test_datasets

        to_evaluate = dataset_config.get('to_evaluate', [])
        to_encode = dataset_config.get('to_encode', [])
        to_decode = dataset_config.get('to_decode', [])
        object_detection_datasets = dataset_config.get('object_detection_datasets', [])
        bosch_datasets = dataset_config.get('bosch_datasets', [])
        
        return (
            base_datasets,
            test_datasets, 
            all_datasets,
            to_evaluate,
            to_encode,
            to_decode,
            object_detection_datasets,
            bosch_datasets
        )

# PATHS
class YamlFiles:
    def __init__(self, yaml_dir):
        self.general_params = os.path.join(yaml_dir, 'param.yaml')
        self.training_param = os.path.join(yaml_dir, 'training_param.yaml')
        self.dataset_names = os.path.join(yaml_dir, 'dataset_names.yaml')
        self.all_scores = os.path.join(yaml_dir, 'all_scores.yaml')
        self.model_info = os.path.join(yaml_dir, 'model_info.yaml')


class Paths:
    def __init__(self, root, src, saves, yaml_dir, **directories):
        self.root = root
        self.src = src
        self.saves = saves
        self.yaml_dir = yaml_dir
        self.yaml_files = YamlFiles(yaml_dir)
        for name, path in directories.items():
            setattr(self, name, path)


def define_paths(use_external_ssd=False,
                 docker_mount=False,
                 print_output=False):

    if 'google.colab' in str(get_python_version()):
        root = '/content/drive/MyDrive'
    elif use_external_ssd and docker_mount:
        root = '/app'
    elif use_external_ssd and not docker_mount:
        root = 'E:/Thesis-Code'
    else:
        root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    src = os.getcwd() if use_external_ssd else os.path.join(root, 'src')
    saves = os.path.join(root, 'saves')
    yaml_dir = os.path.join(src, 'yaml')

    directories = ['datasets', 'datasets_prepared', 'datasets_prepared_whole',
                   'models', 'histories', 'encoded', 'decoded',
                   'output', 'logs']

    paths = {name: os.path.join(saves, name) for name in directories}

    if print_output:
        print(separator)
        print('PATHS:')
        print('Root directory:', root)
        print('Source directory:', src)
        print('Saves directory:', saves)
        for name, path in paths.items():
            print(f'{name.capitalize()}:', path)
        print('YAML dir:', yaml_dir)
        print(separator)

    return Paths(root, src, saves, yaml_dir=yaml_dir, **paths)


# GENERAL PARAMS OF YAML
class GeneralParameters:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def load_parameters_from_yaml(param_yaml_path, print_output=False):
    with open(param_yaml_path, 'r') as file:
        param = yaml.safe_load(file)

    params = GeneralParameters(**param)

    if print_output:
        print(separator)
        print("YAML PARAMETERS:")
        for section, section_params in param.items():
            print(f"\n{section.capitalize()}:")
            for key, value in section_params.items():
                print(f"{key.capitalize()}: {value}")
        print(separator)

    return params


# GPU AVAILABILTY
def set_tf_device(device_type='cpu'):
    print(separator)
    if device_type == 'cpu':
        physical_devices = tf.config.list_physical_devices('CPU')
        tf.config.set_visible_devices(physical_devices, 'CPU')
        tf.config.experimental.set_visible_devices([], 'GPU')

        visible_devices = tf.config.get_visible_devices()
        print("Visible devices after switching to CPU:")
        print(visible_devices)

    elif device_type == 'gpu':
        tf.debugging.set_log_device_placement(False)
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')

        print("Available GPU devices:")
        print(gpu_devices)

        for device in gpu_devices:
            print(f"GPU device: {device.name}:")
            tf.config.experimental.set_memory_growth(device, True)
            tf.config.experimental.VirtualDeviceConfiguration(device)

    else:
        print(f"Invalid device type: {device_type}. Use 'cpu' or 'gpu'")

    print(separator)
    print(f'Switched to {device_type}')

    return device_type


# GET SUBFOLDER PATHS
def get_subfolder_path(models_cat='', sub_cat='',
                       subfolder='', model_name='',
                       encoded=False):
    base_path = os.path.join(
        _paths.encoded) if encoded else os.path.join(
            _paths.models) 

    model_dir = base_path
    if models_cat:
        model_dir = os.path.join(model_dir, models_cat)
    if sub_cat:
        model_dir = os.path.join(model_dir, sub_cat)
    if subfolder:
        model_dir = os.path.join(model_dir, subfolder)
    if model_name:
        model_dir = os.path.join(model_dir, f'{model_name}.h5')

    return model_dir


# PRINT DICT KEYS
def print_dict_content(dict):
    for dataset_name, _ in dict.items():
        print(f'Dataset: {dataset_name}')


# CONFIG
_paths = None
_params = None
_dataset_names = None


def set_paths(use_external_ssd=True, docker_mount=False, print_output=False):
    global _paths
    _paths = define_paths(use_external_ssd, docker_mount, print_output)


def set_params():
    global _params
    paths = get_paths()
    _params = load_parameters_from_yaml(paths.yaml_files.general_params)


def set_dataset_names():
    global _dataset_names
    paths = get_paths()
    _dataset_names = DatasetNames(os.path.join(paths.yaml_files.dataset_names))


def get_paths():
    global _paths
    return _paths


def get_params():
    global _params
    return _params


def get_dataset_names():
    global _dataset_names
    return _dataset_names
