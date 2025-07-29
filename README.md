# Dataset Analysis

---
## Abstract

This work uses a Residual Convolutional Autoencoder (R-CAE) for dimensionality reduction, analysis, and evaluation of complex datasets of different domains and properties. By training on selected domain-representative datasets, the model aims to capture the characteristics of the variant datasets and then drastically reduce their dimensionality using the encoder component. A representative feature vector is derived for each dataset from the acquired dataset encodings, through additional dimensionality reduction via aggregation, compactly encapsulating its key features. The representatives are then evaluated in a subsequent distance-based selection process based on their domain properties and proximity to real data, aiming to identify similar datasets, cluster them, and create an ordinal ranking indicating the general suitability of the datasets for ML tasks. 

---
## How to install

```bash
git clone git@github.com:IRAS-HKA/student_dataset_analysis.git
cd student_dataset_analysis
```

## How to run / use

### Project structure overview

After cloning the git project, the project structure includes the following main directories:

- `docker`: Contains Dockerfile and related scripts for containerizing the project.
- `saves`: Directory for storing saved models, datasets, encoded, decoded data and output 
- `src`: Source code for the project, including scripts and Jupyter notebooks and yaml files

### Preparation

Datasets intended for analysis should be extracted to the `saves/datasets` directory:
```bash
- docker
- saves
-- datasets
--- dataset_1
--- dataset_2
--- dataset_3
...
- src
```

If datasets are saved on external drive, set environment variable in ```.vscode``` ```settings.json``` to save path (folder structure of ```saves``` should be the same as in the git project)
```json
{
    "terminal.integrated.env.windows": {
        "EXTERNAL_DRIVE_PATH": "C:\\path\\to\\saves"
    },
    "terminal.integrated.env.linux": {
        "EXTERNAL_DRIVE_PATH": "/path/to/saves"
    }
}
```

The env can be used to mount the directory ```saves``` containing the extracted datasets within ```/docker/run_docker.sh```:

```
MOUNT_PATH=$EXTERNAL_DRIVE_PATH
```
```bash
-v $MOUNT_PATH:/app/saves \
```

### Build and run docker

```bash
./docker/build_docker.sh

./docker/run_docker.sh
```

### Jupyter Notebooks Overview
The project is structured into separate Jupyter notebook files for different steps of the ML workflow:

#### Config per ipynb
Each ```.ipynb``` file has a first cell containing params to regulate ```device_type``` and saves path using ```use_external_ssd``` and ```docker_mount```.

```python
launch_tensorboard = False
use_external_ssd = False
docker_mount = False
print_output = False
device_type = 'cpu'
```


#### nb_data
1. PNG files were resized and padded 
2. JSON files were used to create a single mask per PNG and then resized to the same size
3. PNG and JSON information was stacked and all datasets were saved of type ```.npy``` in ```saves/datasets_prepared_whole```
4. Prepared datasets were split in ```train```, ```val```, ```test```
5. Rebuild of rgb and mask channels can be tested
6. Includes utilities to test preprocessed image and mask rebuild and for 
data management (e.g., moving train files, renaming model files).


#### nb_visualize
* Used to test and visualize model configurations.


#### nb_model_train
* Requires data prepared in nb_data to create the training data in /saves/datasets_prepared. The folder structure should be as follows:
```bash
saves/datasets_prepared
- dataset_1_prepared
  - train
    - preprocessed_{dataset_name}_{nr}.npy
  - val
  - test
...
```

* Hyperparameters are defined in params.yaml.
* Trains the models configured in src/yaml/training_param.yaml.


```yaml
# Example for empty training config:
model_1:
  model_name: ''
  filters: []
  real_amount: 0
  imgs_to_train_on: 0
  epochs_per_dataset: 0
  dataset_names: []
  autoencoder_type: ''
  dense_units: []
```
* ```autoencoder_type``` can be set to:
    * ```classic``` for Classic AE with dense layer
    * ```cae``` for Convoltuional Autoencoder 
    * ```r_cae``` for Residual Convolutional Autoencoder 

* Depending on the `autoencoder_type`, the `filters` and/or `dense_units` lists should be populated with:
    * The amount of filters to be used in convolutional encoding/decoding steps, with each filter element determining the filter amount of the specific convolutional encoding/decoding step.
    * The amount of units to be used in dense encoding/decoding steps, with each unit element determining the units amount of the specific dense encoding/decoding step.


#### nb_eval
* Evaluates the model trainings using the model paths saved in src/yaml/models_info.yaml.
* Calculates SSI, PSNR, MSE, MAE metrics for all datasets to be evaluated and saves the result in src/yaml/all_scores.yaml.
* Datasets for evaluation are defined in /src/yaml/dataset_names.yaml under "to_encode".
* Plots validation loss of all models.


#### nb_latent
* Encodes the prepared datasets.
* Analyzes the latent space of individual data.
* Aggregates datasets to feature vectors using statistical metrics (mean, median, max, min, sum, variance, standard deviation, skewness).
* Visualizes the latent space of aggregated feature vectors.


#### nb_svm
* Trains an SVM on selected representative feature vectors obtained by encoded datasets.
* Predicts domains for feature vectors.


#### nb_cluster_analysis
* Applies cluster analysis on aggregated feature vectors, feature vectors, and original data using PCA, t-SNE, and UMAP.


#### nb_performance
* Loads the encoded datasets to use as a base for ranking.
* Aggregates them with a given metric.
* Calculates a distance matrix on feature-vector-level and feature-map-level (normalized or not).
* Visualizes distances using MDS and tables.
* Determines domain order by using the distance of base datasets to dataset_sort_by.
* Uses distances to detect domain members (naturally or with given base datasets as references) with a set threshold value.
* Uses domain order, domain membership, and proximity to dataset_sort_by (e.g., for real proximity) to create an ordinal ranking to consider in the final ranking.















