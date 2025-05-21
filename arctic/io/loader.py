import pandas as pd

def read_data(input_file: str, enc: str = "utf-8") -> pd.DataFrame:
    """
    Reads data from a file, filtering only lines that start with 'D'.

    This function skips all lines starting with characters other than 'D'
    and converts the filtered lines into a pandas DataFrame.

    :param input_file: Path to the input file to read
    :type input_file: str
    :param enc: Encoding of the input file, defaults to "utf-8"
    :type enc: str, optional

    :return: DataFrame containing the filtered data
    :rtype: pd.DataFrame
    """
    with open(input_file, "r", encoding=enc) as f:
        filtered_lines = [line for line in f if line.startswith("D")]

    # Convert filtered lines into DataFrame
    from io import StringIO
    df = pd.read_csv(
        StringIO("".join(filtered_lines)),
        delimiter=",",
        low_memory=False
    )
    return df
