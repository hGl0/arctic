import argparse
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import AgglomerativeClustering, KMeans

import arctic

def set_up_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"arctic_analysis_script_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")
    return log_file

import arctic.io.loader
import arctic.io.cleaner
from sklearn.preprocessing import LabelEncoder
def load_and_clean_data(filepath):

    df = arctic.io.loader.read_data(str(filepath))
    arctic.io.cleaner.no_white_space(df)
    # time_col = input("Are there any time columns that should be converted to datetime?: ")
    time_col = 'string'
    time_format = '%d.%m.%Y-%H:%M:%S'
    arctic.io.cleaner.to_date(df, time_col, time_format)

    # encode non-numeric columns
    le = LabelEncoder()
    for col in df.columns:
        pass

    logging.info(f"Loaded data from {filepath} with shape {df.shape}")
    logging.info(f"Info about DataFrame:\n{df.info()}")

    return df

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
def select_scaler(scaler):
    if scaler == 'StandardScaler': return StandardScaler()
    if scaler == 'RobustScaler': return RobustScaler()
    if scaler == 'MinMaxScaler': return MinMaxScaler()
    else: return None
    # if isinstance(scaler, callable): return scaler

def scale_and_plot(df, scaler, output_path):
    # scale data
    time_col = 'string'

    constraints = input("Do you have any time constraints? (y/n): ")
    if constraints == 'y':
        start = input("Which start date would you like to select?\n "
                      "Only data after the date will be considered: ")
        df = df[(df[time_col] >= start)]
        end = input("Which end date would you like to select?\n "
                    "Only data before the date will be considered: ")
        df = df[(df[time_col] <= end)]
        interest_time = input("Are you interested in a certain time range (e.g. DJFM)? (y/n): ")
        if interest_time == 'y':
            interest_time = input("Which time range are you interested in?\n"
                                  "Please provide the number of the months as a comma-separated list: ").split(',')
            df = df[df[time_col].dt.month.isin(interest_time)]

    scaler = select_scaler(scaler)
    if scaler is None:
        logging.info(f"Scaling data with {scaler} and saved comparison plot at {output_path}")
        return df

    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    features = input("Please give a comma-separated list of features to include in the plot: ").split(',')
    num_subplots = len(features)/2

    fig, axes = plt.subplots(num_subplots, figsize=(10, 3*num_subplots))
    axes = axes.flatten()
    for i in range(len(axes)):
        axes[i].plot(df_scaled[features[i]], label=features[i])
        axes[i].plot(df_scaled[features[i+1]], label=features[i+1])
        axes[i].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,"scaled_features.png"))
    plt.close()

    logging.info(f"Scaling data with {scaler} and saved comparison plot at {output_path}")
    return df_scaled


def filter_data(df, features, filter, output_path):
    # detect seasonality ask for confirmation in each feature or get argument "automate" to do as deemed suitable
    # if yes: filter with ssa/eeof/yoy
    # plot filtered vs. unfiltered
    logging.info(f"Filtering data with {filter} and saved comparison plot at {output_path}")
    # return filtered

def run_pca(df, output_path):
    x_new, scores, pca = arctic.compute_pca(df, n_comp=2)
    arctic.plot_pca(pca, x_new, output_path)
    logging.info(f"PCA computed and ploted.")


def detect_opt_clusters(data, method, output_path, max_k):
    if method == 'gap statistic':
        result = arctic.gap_statistic(data, max_k=max_k)

    plt.plot(np.arange(len(result)), result[0])
    plt.savefig(output_path)

    k_opt = -1

    for k in range(len(result)):
        if result[k][0] > result[k+1][0] - result[k][1]:
            k_opt = k
    logging.info(f"Optimal number of clusters {k_opt} with {method} detected. Comparison plot at {output_path}.")
    return k_opt

def select_model(model_name, n_clusters=None):
    if model_name == 'kmeans':
        return KMeans(n_clusters=n_clusters or 3)
    if model_name == 'hierarchical':
        return AgglomerativeClustering(n_clusters=n_clusters or 3, linkage='average')
    raise ValueError(f"Unknown model: {model_name}")

def cluster_distribution(df, model_name, features, output_path, n_clusters=None):
    # cluster with model
    # get features
    # plot distribution of features per found class
    # aggregation per class
    logging.info(f"Clustering data with {model_name} using {features}. Stored distribution of classes at {output_path}")

def main():
    """Parse arguments and call processing function."""
    parser = argparse.ArgumentParser(description="Process a CSV file using arctic.")
    parser.add_argument("input", type=str, help="Path to input CSV file")
    parser.add_argument("output_path", type=str, help="Path to the directory, where the processed output should be saved")
    parser.add_argument("model", type=str, nargs='?',
                        default="hierarchical", choices=['hierarchical', 'kmeans'],
                        help="Clustering model: hierarchical (default), kmeans")
    parser.add_argument('other', type=str, choices=['verbose'],
                        nargs='?', default=None, help='Additional parameters')

    args = parser.parse_args()

    # Convert relative paths to absolute paths
    input_file = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output_path)
    model_name = args.model

    # set up logs
    set_up_logging(log_dir='logs')
    # get data
    df = load_and_clean_data(input_file)

    # scaling
    scaler = input("Would you like to scale the data? (y/n): ").lower()
    if scaler == 'y':
        scaler = input("What scaler would you like to use? (Standard/MinMax/Robust): ").lower()
        scale_and_plot(df, scaler, output_path)

    # filtering
    filter = input("Would you like to filter the data? (y/n): ").lower()
    if filter == 'y':
        filter = input("How would you like to filter? (EEOF/SSA/YoY/automated): ")
        features = input("Comma-separated list of features to use: ").split(',')
        filter_data(df, features, filter, output_path)

    # determine optimal k
    model = select_model(model_name)
    k_opt = input("Would you like to determine an optimal k? (y/n): ").lower()
    if k_opt == 'y':
        method = input("Which method would you like to use? (elbow method/gap statistic/silhouette): ").lower()
        k_max = input("How many clusters would you like to use at most?: ")
        k_opt = detect_opt_clusters(df, model, method, output_path, max_k=k_max)
    else:
        k_opt = input("Please input the desired number of clusters: ")

    # cluster data
    logging.info(f"Start clustering with {model} and k={k_opt}")
    features = input("Would you like to use only certain features? (y/n): ").lower()
    if features == 'y':
        features = input("Comma-separated list of features to use: ").split(',')
    else:
        features = None
    cluster_distribution(df, model, features, output_path, k_opt)

    # interpretation and aggregation per class
    # comparison to designated class column

if __name__ == "__main__":
    main()


