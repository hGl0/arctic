# This is an example script that processes input data
# and plots graphs and insight into an output directory

import argparse
import pandas as pd
import os
import src.arctic as arctic


# multiple arguments for other possible...!! First version!!
def process_data(input_file, output_path, model, other):
    """Main function to process data."""
    # Read input CSV
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            filtered_lines = [line for line in f if line.startswith("D")]

        # Convert filtered lines into DataFrame
        from io import StringIO
        df = pd.read_csv(
            StringIO("".join(filtered_lines)),
            delimiter=",",
            low_memory=False
        )

        # To Do: include other parameters nicely
        if other:
            print('Successfully read input data')

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return


    # insert work flow later!
    # To Do:
    # csv with most important features by pca
    # 2D/3D scatter plot with pca vectores
    # correlation matrix as csv or png? both?

    # spider plot with 6 most important features
    # hierarchical: dendrogram
    # kmeans: 2D/3D scatter with colored clusters? Geo map with colored clusters?

    print(f'Magic data processing using {model}.')


    # Save the processed file
    processed_df = df
    processed_df.to_csv(os.path.join(output_path, "processed_data.csv"), index=False)
    print(f"Processed data saved to {output_path}")


def main(input_file, output, model, other):
    try:
        process_data(input_file, output, model, other)
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

    # Call processing function
    main(input_file, output_path, args.model, args.other)

