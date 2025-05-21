# This is an example script that processes input data
# and plots graphs and insight into an output directory

import argparse
import os
import arctic as arctic
from sklearn.cluster import AgglomerativeClustering


# multiple arguments for other possible...!! First version!!
def process_data(input_file, output_path, model):
    """Main function to process data."""
    # Read input CSV
    try:
        data = arctic.read_data(input_file)
        print("Successfully read data.")
    except Exception as e:
        raise Exception(f"Error while reading CSV file: {e}")

    # Correlation
    try:
        arctic.visualization.plot.plot_correlation(data,
                                                   savecsv=f"{output_path}/corr_results.csv",
                                                   savefig=f"{output_path}/corr_plot.png")
        print("Successfully plotted correlation matrix.")
    except Exception as e:
        raise Exception(f"Error while plotting correlation matrix: {e}")

    # PCA
    # does not work yet!
    try:
        pca = arctic.compute_pca(data.select_dtypes('number'), comp=data.shape[1],
                                 savecsv=f"{output_path}/pca_results.csv",
                                 plot_type='2D', savefig=f"{output_path}/pca_2D.png")
        print("Successfully compute the PCA:")
    except Exception as e:
        raise Exception(f"Error while computing PCA: {e}")

    # insert work flow later!
    # To Do:

    # spider plot with 6 most important features
    # hierarchical: dendrogram
    # kmeans: 2D/3D scatter with colored clusters? Geo map with colored clusters?

    print(f'Magic data processing using {model}.')


    # Save the processed file
    # processed_df = df
    # processed_df.to_csv(os.path.join(output_path, "processed_data.csv"), index=False)
    # print(f"Processed data saved to {output_path}")


def main(input_file, output, model, **kwargs):
    try:
        process_data(input_file, output, model, **kwargs)
    except Exception as e:
        print(f"Error {e} while executing the script")


if __name__ == "__main__":
    """Parse arguments and call processing function."""
    parser = argparse.ArgumentParser(description="Process a CSV file using arctic.")
    parser.add_argument("input", type=str, help="Path to input CSV file")
    parser.add_argument("output_path", type=str, help="Path to save the processed output")
    parser.add_argument("model", type=str, nargs='?',
                        default="hierarchical", choices=['hierarchical', 'kmeans'],
                        help="Clustering model: hierarchical (default), kmeans")
    parser.add_argument('other', type=str, choices=['verbose'],
                        nargs='?', default=None, help='Additional parameters')

    args = parser.parse_args()

    # Convert relative paths to absolute paths
    input_file = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output_path)

    if args.model == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=3, linkage='average')

    # Call processing function
    main(input_file, output_path, model)


