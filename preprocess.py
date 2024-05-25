import numpy as np
import pandas as pd
import json
from io import StringIO
import sklearn.model_selection
import torch

test = False

# Function to parse the Volumes column
def parse_volumes(volumes_str):
    # Replace single quotes with double quotes for proper JSON parsing
    volumes_str = volumes_str.replace("'", "\"")
    # Load the string as JSON
    volumes_json = json.loads(volumes_str)
    return {item['Key']: item['Value'] for item in volumes_json}


# Function to normalize the data
def normalize_data(df):
    # Normalize the data
    df_normalized = (df - df.max()) / df.std()
    return df_normalized

def create_datasplit(df):
    # Split the data into training and test sets
    random_state = 33
    train_df, test_df = sklearn.model_selection.train_test_split(df, test_size=0.2, random_state=random_state)
    test_df, val_df = sklearn.model_selection.train_test_split(test_df, test_size=0.5, random_state=random_state)
    return train_df, test_df, val_df


def compute_simultaneity(user1_timeries: np.ndarray, user2_timeries: np.ndarray) -> np.ndarray:
    """
    Compute the simultaneity between two users based on their time series data.
    Args:
        user1_timeries (np.ndarray): Time series data for user 1
        user2_timeries (np.ndarray): Time series data for user 2
    Returns:
        np.ndarray: Simultaneity score for user 1

    >>> u1 = np.array([4,2,4])
    >>> u2 = np.array([0,50,50])
    >>> compute_simultaneity(u1, u2)
    0.6
    >>> np.allclose(compute_simultaneity(u2, u1), 0.06)
    True
    """
    # Compute the correlation between the two time series
    sum_user1 = np.sum(user1_timeries)
    sum_of_max = np.sum(np.maximum(user1_timeries - user2_timeries, 0))
    simultaneity_score = 1 - sum_of_max/sum_user1
    return simultaneity_score


def compute_and_store_simultaneity_scores(df: pd.DataFrame):
    pass


if __name__ == '__main__':
    if test:
        import doctest
        doctest.testmod()

    # Read the CSV data
    # filename = StringIO(data)
    filename = 'dataset/consumption-1y.csv'
    df = pd.read_csv(filename, sep=';', usecols=["Id", "Volumes"])

    # Apply the function to the Volumes column
    df['Volumes'] = df['Volumes'].apply(parse_volumes)

    # Create a DataFrame from the 'Volumes' dictionaries
    volumes_df = df['Volumes'].apply(pd.Series)

    # Concatenate the original DataFrame with the new DataFrame of volumes
    result_df = pd.concat([df.drop(columns=['Volumes']), volumes_df], axis=1)
    result_df.to_csv('dataset/consumption-1y-processed.csv', sep=';', index=False)

    # # Print the final DataFrame
    # print(result_df.info())
    # print(result_df.head())

    # _, _, _ = create_datasplit(result_df)
