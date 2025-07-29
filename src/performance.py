from matplotlib import pyplot as plt
import latent
import numpy as np
from scipy.stats import skew
import pandas as pd
from IPython.display import display
from sklearn.manifold import MDS
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.preprocessing import QuantileTransformer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import copy

'''
AGGREGATION
'''


# AGGREGATE FEATURE VECTORS
def aggregate_feature_vectors(encoded_datasets, axis):
    aggregated_datasets = {}

    for dataset_name, encoded_data in encoded_datasets.items():
        mean_values = np.mean(encoded_data, axis)
        median_values = np.median(encoded_data, axis)
        std_dev_values = np.std(encoded_data, axis)
        max_values = np.max(encoded_data, axis)
        min_values = np.min(encoded_data, axis)
        sum_values = np.sum(encoded_data, axis)
        variance_values = np.var(encoded_data, axis)
        skewness_values = skew(encoded_data, axis, nan_policy='omit')

        aggregated_datasets[dataset_name] = {
            'mean': mean_values,
            'median': median_values,
            'std_dev': std_dev_values,
            'max': max_values,
            'min': min_values,
            'sum': sum_values,
            'variance': variance_values,
            'skewness': skewness_values

        }

    return aggregated_datasets


# GET AGGREGATED DATA
def get_aggregated_data(model_name, models_folder,
                        base_dataset_names, test_dataset_names):

    encoded_base_datasets = latent.load_encoded_datasets(
        base_dataset_names, model_name, models_folder)
    encoded_test_datasets = latent.load_encoded_datasets(
        test_dataset_names, model_name, models_folder)

    aggregated_feature_vectors_base = aggregate_feature_vectors(
        encoded_base_datasets, axis=0)
    aggregated_feature_vectors_test = aggregate_feature_vectors(
        encoded_test_datasets, axis=0)

    all_aggregated_feature_vectors = {}
    all_aggregated_feature_vectors.update(aggregated_feature_vectors_base)
    all_aggregated_feature_vectors.update(aggregated_feature_vectors_test)

    return (
        aggregated_feature_vectors_base,
        aggregated_feature_vectors_test,
        all_aggregated_feature_vectors
    )


# SCALER
def get_scaler(scaler_name):
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'minmax': MinMaxScaler(feature_range=(0, 1)),
        'normalizer': Normalizer(),
        'quantile': QuantileTransformer(output_distribution='uniform')
    }

    return scalers.get(scaler_name.lower(), None)


# NORMALIZE
def normalize_aggregated_feature_vectors(aggregated_feature_vectors, scaler):
    normalized_aggregated_feature_vectors = {}

    for dataset_name, aggregates in aggregated_feature_vectors.items():
        normalized_aggregates = {}
        for aggregate_type, feature_vector in aggregates.items():
            original_shape = feature_vector.shape
            reshaped_vector = feature_vector.reshape(-1, 1) if len(
                original_shape) > 1 else feature_vector

            scaled_vector = scaler.fit_transform(reshaped_vector)

            if len(original_shape) > 1:
                scaled_vector = scaled_vector.reshape(original_shape)

            normalized_aggregates[aggregate_type] = scaled_vector
        normalized_aggregated_feature_vectors[
            dataset_name] = normalized_aggregates

    return normalized_aggregated_feature_vectors


'''
CALCULATE DISTANCES
'''


# CALCULATE EUCLIDEAN DISTANCE
def calculate_euclidean_distances(test_feature, base_feature):
    test_feature = test_feature.flatten(
        ).reshape(-1, 1).reshape(test_feature.shape)
    base_feature = base_feature.flatten(
        ).reshape(-1, 1).reshape(base_feature.shape)
    return np.sqrt(np.sum((test_feature - base_feature) ** 2))


# CALCULATE MANHATTAN DISTANCE
def calculate_manhattan_distances(test_feature, base_feature):
    test_feature = test_feature.flatten(
        ).reshape(-1, 1).reshape(test_feature.shape)
    base_feature = base_feature.flatten(
        ).reshape(-1, 1).reshape(base_feature.shape)
    return np.sum(np.abs(test_feature - base_feature))


# CALCULATE DISTANCES ON FEATURE LEVEL
def calculate_distances_on_feature_level(test_dataset,
                                         base_datasets,
                                         aggregate_type,
                                         distance_type):
    feature_distances = {}

    for feature_index in range(test_dataset[aggregate_type].shape[2]):
        test_feature = test_dataset[aggregate_type][:, :, feature_index]
        feature_distances[feature_index] = {}

        for base_name, base_dataset in base_datasets.items():
            base_feature = base_dataset[aggregate_type][:, :, feature_index]

            if distance_type == 'euclidean':
                distance = calculate_euclidean_distances(
                    test_feature, base_feature)
            if distance_type == 'manhattan':
                distance = calculate_manhattan_distances(
                    test_feature, base_feature)

            feature_distances[feature_index][base_name] = distance

    feature_distances_df = pd.DataFrame.from_dict(feature_distances,
                                                  orient='index')
    return feature_distances_df


# CALCULATE DISTANCES ON DATASET LEVEL
def calculate_distances_on_dataset_level(test_dataset,
                                         base_datasets,
                                         aggregate_type,
                                         distance_type):
    distances = {}

    for base_name, base_dataset in base_datasets.items():

        if distance_type == 'euclidean':
            distance = calculate_euclidean_distances(
                test_dataset[aggregate_type],
                base_dataset[aggregate_type])
        if distance_type == 'manhattan':
            distance = calculate_manhattan_distances(
                    test_dataset[aggregate_type],
                    base_dataset[aggregate_type])

        distances[base_name] = distance

    distances_df = pd.DataFrame({'dataset': list(distances.keys()),
                                 distance_type: list(distances.values())})
    return distances_df


# CALCULATE ALL DISTANCES
def calculate_distances(aggregated_data, base_data, distance_type):
    all_distances = {dataset_name: {} for dataset_name in aggregated_data}
    all_feature_distances = {
        dataset_name: {} for dataset_name in aggregated_data}

    for dataset_name in aggregated_data:
        aggregated_dataset = aggregated_data[dataset_name]

        for aggregate_type in aggregated_dataset:
            dataset_distance_df = calculate_distances_on_dataset_level(
                aggregated_dataset, base_data, aggregate_type,
                distance_type)

            dataset_distance_df[
                distance_type] = dataset_distance_df[distance_type].apply(
                lambda x: x[0][0] if isinstance(x, (list, np.ndarray)) else x)

            feature_distances_df = calculate_distances_on_feature_level(
                aggregated_dataset, base_data, aggregate_type,
                distance_type)

            if distance_type == 'cosine':
                feature_distances_df = feature_distances_df.map(
                    lambda x: x[0][0])

            all_distances[dataset_name][aggregate_type] = dataset_distance_df
            all_feature_distances[dataset_name][
                aggregate_type] = feature_distances_df

    return all_distances, all_feature_distances


# GET DISTANCES BETWEEN DATASET LIST X AND LIST Y
def get_distances_data(aggregated_feature_vectors_test,
                       aggregated_feature_vectors_base,
                       distance_type):

    all_distances_base_ds, all_feature_distances_base_ds = calculate_distances(
        aggregated_feature_vectors_base, aggregated_feature_vectors_base,
        distance_type)

    all_distances_test_ds, all_feature_distances_test_ds = calculate_distances(
        aggregated_feature_vectors_test, aggregated_feature_vectors_base,
        distance_type)

    all_distances_datasets = {}
    all_distances_datasets.update(all_distances_base_ds)
    all_distances_datasets.update(all_distances_test_ds)

    all_distances_features = {}
    all_distances_features.update(all_feature_distances_base_ds)
    all_distances_features.update(all_feature_distances_test_ds)

    return all_distances_datasets, all_distances_features


'''
DISTANCE MATRIX
'''


# MATRIX NORMALIZATION
def normalize_matrix(matrix):
    min_val = np.min(matrix[np.nonzero(matrix)])
    max_val = np.max(matrix)
    range_val = max_val - min_val
    normalized_matrix = (matrix - min_val) / range_val
    np.fill_diagonal(normalized_matrix, 0)
    return normalized_matrix


# VISUALIZE SIMILARITY MATRIX
def visualize_similarity_matrix(distances_matrix, dataset_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(distances_matrix, annot=True, fmt=".2f",
                cmap="YlGnBu", xticklabels=dataset_names,
                yticklabels=dataset_names)
    plt.title('Pairwise Similarity Between Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Dataset')
    plt.show()


# FEATURE MAP MATRIX
def create_fm_distance_matrix(all_to_all_distances_features,
                              feature_index, metric):
    dataset_names = list(all_to_all_distances_features.keys())
    num_datasets = len(dataset_names)
    distances_matrix = np.zeros((num_datasets, num_datasets))

    for i, dataset_i in enumerate(dataset_names):
        for j, dataset_j in enumerate(dataset_names):
            if dataset_i == dataset_j:
                distances_matrix[i, j] = 0
            else:
                try:
                    distance = all_to_all_distances_features[
                        dataset_i][metric].loc[feature_index, dataset_j]
                    distances_matrix[i, j] = distance
                except KeyError:
                    print(f'Distance from {dataset_i} to {dataset_j}',
                          f'for feature {feature_index} not found.')

    return distances_matrix, dataset_names


# FEATURE MAP MATRIX DICT
def create_fm_distance_matrix_dict_norm(fm_distances, num_features, metric):
    normalized_matrix_dict = {}
    for feature_index in range(num_features):
        key = feature_index
        matrix, _ = create_fm_distance_matrix(fm_distances,
                                              feature_index,
                                              metric)
        normalized_matrix = normalize_matrix(matrix)
        normalized_matrix_dict[key] = normalized_matrix
    return normalized_matrix_dict


def create_fm_distance_matrix_dict(fm_distances, num_features, metric):
    matrix_dict = {}
    for feature_index in range(num_features):
        key = feature_index
        matrix, _ = create_fm_distance_matrix(fm_distances,
                                              feature_index,
                                              metric)
        matrix_dict[key] = matrix
    return matrix_dict


# FEATURE VECTOR MATRIX
def create_fv_distance_matrix(all_distances_data,
                              distance_type,
                              metric):
    first_key = next(iter(all_distances_data))
    dataset_names = all_distances_data[first_key][
        metric]['dataset'].tolist()

    name_to_index = {name: idx for idx, name in enumerate(dataset_names)}

    num_datasets = len(dataset_names)
    distances_matrix = np.zeros((num_datasets, num_datasets))

    for dataset, distances in all_distances_data.items():
        i = name_to_index[dataset]
        for _, row in distances[metric].iterrows():
            j = name_to_index[row['dataset']]
            distances_matrix[i, j] = row[distance_type]
            distances_matrix[j, i] = row[distance_type]

    return distances_matrix, dataset_names


'''
DISPLAY DISTANCES
'''


# DISTANCES ON FEATURE VECTOR
def create_distance_table(feature_vector_distances, aggregate_type,
                          distance_type, base_order=None,
                          sort_by='adidas_real'):
    target_datasets = base_order if base_order is not None else list(
        feature_vector_distances.keys())
    distances = {dataset: {} for dataset in feature_vector_distances.keys()}

    for dataset1 in feature_vector_distances.keys():
        for dataset2 in target_datasets:
            if dataset1 != dataset2:
                row = feature_vector_distances[dataset1][aggregate_type]
                distance = row.loc[row[
                    'dataset'] == dataset2, distance_type].values[0]
                distances[dataset1][dataset2] = distance
            else:
                distances[dataset1][dataset2] = 0.0

    distance_df = pd.DataFrame(distances).T
    distance_df = distance_df[target_datasets]

    sorted_distance_table = distance_df.sort_values(by=sort_by,
                                                    ascending=True)

    return sorted_distance_table


# DISPLAY DISTANCES FEATURE MAPS
def display_distances(dataset_distances, dataset_name, aggregate_type):
    print('***********************************************')
    print(f'Aggregate type: {aggregate_type}')
    print(f'Distances for dataset: {dataset_name}')
    display(dataset_distances[dataset_name][aggregate_type])


# FILTER FOR DATASETS
def filter_distances(distance_df, base_datasets):
    if not set(base_datasets).issubset(distance_df.columns):
        missing = list(set(base_datasets) - set(distance_df.columns))
        raise ValueError(f'MIssing datasets: {missing}')

    filtered_df = distance_df[base_datasets]
    return filtered_df


# DISPLAY DISTANCE TABLE
def display_distance_table(dataset_names, matrix, title):
    print(title)
    distance_df = pd.DataFrame(
        matrix,
        index=dataset_names,
        columns=dataset_names
    )

    distance_df = distance_df.map(lambda x: round(x * 1, 4))
    return distance_df


# MDS FEATURE VECTOR
def visualize_fv_similarity(distances_matrix, dataset_names):
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(distances_matrix)

    plt.figure(figsize=(8, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1],
                c='blue', edgecolors='black')
    for i, dataset_name in enumerate(dataset_names):
        plt.annotate(dataset_name, 
                     (coordinates[i, 0], coordinates[i, 1]),
                     xytext=(5, -3),  
                     textcoords='offset points') 

    plt.title('Similarity Visualization of Aggregated Feature Vectors')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.show()

    visualize_similarity_matrix(distances_matrix, dataset_names)


# MDS FEATURE MAP
def visualize_fm_similarity(distances_matrix, dataset_names, feature_index):
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(distances_matrix)

    plt.figure(figsize=(8, 6))
    plt.scatter(coordinates[:, 0],
                coordinates[:, 1],
                c='blue',
                edgecolors='black')
    for i, dataset_name in enumerate(dataset_names):
        plt.annotate(dataset_name, (coordinates[i, 0], coordinates[i, 1]))

    plt.title(f'Similarity Visualization of Feature Map {feature_index}')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.grid(True)
    plt.show()

    visualize_similarity_matrix(distances_matrix, dataset_names)


'''
DOMAIN
'''


# DETECT DOMAINS ON BASE DATASETS
def detect_domains_on_base(distance_df,
                           base_datasets,
                           threshold):
    domain_assignments = {}

    for index, row in distance_df.iterrows():
        distances_to_bases = row[base_datasets]
        closest_base = distances_to_bases.idxmin()
        min_distance = distances_to_bases.min()

        if min_distance <= threshold:
            domain_assignments[index] = closest_base
        else:
            domain_assignments[index] = "undefined"

    domains_df = pd.DataFrame(list(domain_assignments.items()),
                              columns=["Dataset", "Domain"])
    return domains_df


# BUILD DOMAINS NATURALLY
def detect_domains_naturally(distance_df, threshold):
    distances_matrix = distance_df.to_numpy()

    np.fill_diagonal(distances_matrix, 0)

    condensed_distances = squareform(distances_matrix)

    Z = linkage(condensed_distances, method='average')
    clusters = fcluster(Z, t=threshold, criterion='distance')

    dataset_names = distance_df.index.tolist()
    domain_assignments = {dataset: domain for dataset,
                          domain in zip(dataset_names, clusters)}
    domains_df = pd.DataFrame(list(domain_assignments.items()),
                              columns=["Dataset", "Domain"])

    return domains_df


# DOMAIN OVERVIEWS
def create_domain_distance_overviews(distance_df,
                                     domain_mappings,
                                     base_datasets,
                                     show_to_base=False,
                                     show_domains=False):
    domain_overviews = {}

    for domain, group in domain_mappings.groupby('Domain')['Dataset']:
        if show_to_base:
            filtered_df = distance_df.loc[group.values, base_datasets]
        else:
            filtered_df = distance_df.loc[group.values, group.values]

        domain_overviews[domain] = filtered_df

        if show_domains:
            print(f"Domain: {domain}")
            display(filtered_df)

    return domain_overviews


'''
DOMAIN BASED RANKING
'''


# ASSIGN DOMAIN RANKS
def assign_domain_ranks(domain_mappings, domain_order):
    ranks = {domain: rank for rank, domain in enumerate(domain_order, start=1)}
    domain_mappings['Rank'] = domain_mappings['Domain'].map(ranks).fillna(0)
    return domain_mappings


# CREATE DOMAIN BASED RANKING
def create_domain_based_ranking(rank_assignment_df, distances_df, sort_by):
    domain_ranking_df = copy.deepcopy(rank_assignment_df)

    domain_ranking_df['Rank In Domain'] = 0
    domain_ranking_df[f'Distance To {sort_by}'] = 0

    for domain, group in domain_ranking_df.groupby('Domain'):
        if domain == 'undefined' or domain not in distances_df.columns:
            continue

        domain_datasets = group['Dataset']
        domain_distances = distances_df.loc[domain_datasets, sort_by]

        domain_distances_sorted = domain_distances.sort_values()

        for i, (index, distance) in enumerate(domain_distances_sorted.items(),
                                              start=1):
            domain_ranking_df.loc[domain_ranking_df[
                'Dataset'] == index, 'Rank In Domain'] = i
            domain_ranking_df.loc[domain_ranking_df[
                'Dataset'] == index, f'Distance To {sort_by}'] = distance

    filtered_domain_ranking_df = domain_ranking_df[
        domain_ranking_df['Domain'] != 'undefined']
    filtered_domain_ranking_df = filtered_domain_ranking_df.reset_index(
        drop=True)

    return filtered_domain_ranking_df


'''
VARIANCE-PENALTY-REWARD BASED RANKING
'''


# CREATE FM DICT BASED ON METRIC
def calculate_fm_metric(aggregated_data, metric='variance'):
    fm_metric_dict = {}
    for dataset_name, metrics in aggregated_data.items():
        fm_metric_dict[dataset_name] = {}

        metric_vector = metrics[metric] 
        aggregated_metric_fm = np.var(metric_vector, axis=(0, 1)) 

        for feature_index, metric_value in enumerate(aggregated_metric_fm):
            fm_metric_dict[dataset_name][feature_index] = metric_value

    return fm_metric_dict


# CREATE FM IMPORTANCE RANKING PER DATASET
def create_fm_importance_ranking(fm_metric_dict):
    fm_importance_ranking_dict = {}

    for dataset, feature_importances in fm_metric_dict.items():
        sorted_feature_indices = sorted(
            feature_importances, key=feature_importances.get, reverse=True)
        fm_importance_ranking_dict[dataset] = {feature_index: rank for rank,
                                               feature_index in enumerate(
                                                   sorted_feature_indices,
                                                   start=1)}

    return fm_importance_ranking_dict


# CREATE FM IMPORTANCE RANKING DF
def create_fm_importance_ranking_df(fm_importance_ranking):
    fm_indexes = list(fm_importance_ranking[
        next(iter(fm_importance_ranking))].keys())
    fm_importance_ranking_df = pd.DataFrame(
        index=fm_indexes,
        columns=fm_importance_ranking.keys())

    for dataset, rankings in fm_importance_ranking.items():
        for feature_index, rank in rankings.items():
            fm_importance_ranking_df.loc[feature_index, dataset] = rank

    fm_importance_ranking_df.sort_index(inplace=True)

    return fm_importance_ranking_df


# FM IMPORTANCE BASED RANKING
def fm_importance_based_ranking(domain_ranking_df,
                                fm_metric_dict,
                                fm_distance_matrix_dict):
    score_title = 'Importance Weighted Score'
    fm_importance_ranking_df = copy.deepcopy(domain_ranking_df)

    fm_importance_ranking_df[score_title] = 0.0

    for i, row in fm_importance_ranking_df.iterrows():
        dataset_name = row['Dataset']
        total_weighted_distance = 0.0

        for feature_index in range(32): 
            importance_rank = fm_metric_dict[dataset_name][feature_index]
            distances_matrix = fm_distance_matrix_dict[feature_index]
            real_distance = distances_matrix[i][0]  

            weight = 1 / (importance_rank if importance_rank > 0 else 1)
            weighted_distance = real_distance * weight
            total_weighted_distance += weighted_distance

        fm_importance_ranking_df.at[i, score_title] += total_weighted_distance

    fm_importance_ranking_df.sort_values(by=score_title, inplace=True)

    return fm_importance_ranking_df


# CREATE RANK DICT
def create_fm_rank_dict(feature_indexes):
    rank_dict = {}
    for rank, index in enumerate(feature_indexes, start=1):
        rank_dict[index] = rank
    return rank_dict


# CALCULATE REWARD-PENALTY SCORES
def calculate_reward_penalty_scores(fm_matrix, domain_mappings,
                                    reward_ds, penalty_ds, dataset_names,
                                    initial_scores, fm_importance_ranking,
                                    weight=1):
    domain_mappings['Score'] = domain_mappings['Domain'].map(initial_scores)
    reward_index = dataset_names.index(reward_ds)
    penalty_index = dataset_names.index(penalty_ds)

    for feature_index in fm_importance_ranking:
        distances_matrix = fm_matrix[feature_index]
        importance_weight = 1 / (fm_importance_ranking.get(feature_index, 0))

        for i, row in domain_mappings.iterrows():
            real_distance = distances_matrix[i, reward_index]
            dr_distance = distances_matrix[i, penalty_index]

            if real_distance < dr_distance:
                reward = (
                    dr_distance - real_distance) * weight * importance_weight
                domain_mappings.at[i, 'Score'] += reward
            else:
                penalty = (
                    real_distance - dr_distance) * weight * importance_weight
                domain_mappings.at[i, 'Score'] -= penalty

    return domain_mappings


# DETERMINE RANK SORT AND ORDER
def determine_rank_sort_and_ascending(domain_consideration, real_similarity,
                                      fm_importance, sort_by):
    rank_sort = []
    ascending = []

    if domain_consideration:
        rank_sort.append('Rank')
        ascending.append(True)

    if real_similarity:
        rank_sort.append(f'Distance To {sort_by}')
        ascending.append(True)

    if fm_importance:
        rank_sort.append('Score')
        ascending.append(False)

    return rank_sort, ascending


def display_sorted_final_ranking(ranking_df, domain_consideration,
                                 real_similarity,
                                 fm_importance,
                                 sort_by):
    temp_df = copy.deepcopy(ranking_df)
    pd.options.display.float_format = '{:.4f}'.format
    rank_sort, ascending = determine_rank_sort_and_ascending(
        domain_consideration,
        real_similarity,
        fm_importance,
        sort_by)

    print("Rank sort criteria:", rank_sort)
    print("Ascending criteria:", ascending)
    
    temp_df.sort_values(by=rank_sort, ascending=ascending, inplace=True)
    return temp_df
