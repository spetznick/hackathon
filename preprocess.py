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
    """Create a train, test, and validation split of the data.

    Args:
        df (pd.DataFrame): DataFrame containing the data
    Returns:
        pd.DataFrame: DataFrame containing the training data
        pd.DataFrame: DataFrame containing the test data
        pd.DataFrame: DataFrame containing the validation data
    """
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

def compute_joint_score(user1_timeries: np.ndarray, user2_timeries: np.ndarray) -> np.ndarray:
    """Compute the simultaneity scores between two users based on their time series data.
    """
    score1 = compute_simultaneity(user1_timeries, user2_timeries)
    score2 = compute_simultaneity(user2_timeries, user1_timeries)
    joint_score = score1 + score2 - np.abs(score1 - score2)
    return joint_score


def compute_and_store_simultaneity_scores(df_consumers: pd.DataFrame, df_producers: pd.DataFrame, filename=None) -> pd.DataFrame:
    """Compute the simultaneity scores between consumers and producers based on their time series data.

    The consumer to producer score is computed by
        1- sum(max(consumer_timeseries - producer_timeseries, 0)) / sum(consumer_timeseries).
    Flipping the roles yields the second score.
    The joint score is computed based on the two scores by
        score_1 + scores_2 - abs(score_1 - score_2).

    If a filename is provided, the output DataFrame is stored as a CSV file.

    The output DataFrame contains the
        consumerid, producerid, scoreC2P, scoreP2C, jointScore

    Args:
        df_consumers (pd.DataFrame): DataFrame containing the consumers' timeries
        df_producers (pd.DataFrame): DataFrame containing the consumers' timeries
    Returns:
        pd.DataFrame: DataFrame containing the simultaneity scores where each row
        contains the consumerid, producerid, score consumer to producer, score producer to consumer, joint score
    """
    # Initialize the DataFrame to store the simultaneity scores
    entries = []

    # Iterate over the consumers and producers to compute the simultaneity scores
    for consumer_id, consumer in df_consumers.iterrows():
        for producer_id, producer in df_producers.iterrows():
            if producer_id == consumer_id:
                continue
            # Compute the simultaneity scores
            score_c2p = compute_simultaneity(consumer, producer)
            score_p2c = compute_simultaneity(producer, consumer)
            joint_score = score_c2p + score_p2c - np.abs(score_c2p - score_p2c)

            # Store the scores in the DataFrame
            entries.append({'consumerid': consumer_id, 'producerid': producer_id,
                                                      'scoreC2P': score_c2p, 'scoreP2C': score_p2c, 'jointScore': joint_score})

    df_simultaneity = pd.DataFrame(entries)
    if filename:
        df_simultaneity.to_csv(filename, index=False)
    return df_simultaneity


def best_matches(ids: list, scores: pd.DataFrame) -> pd.DataFrame:
    """Find the best matches for each consumer based on the joint score.

    Args:
        ids (list): List of consumer ids
        scores (pd.DataFrame): DataFrame containing the simultaneity scores
    Returns:
        pd.DataFrame: DataFrame containing the best matches for each consumer
    """
    # Initialize the DataFrame to store the best matches
    best_matches = []

    # Iterate over the consumer ids to find the best matches
    for consumer_id in ids:
        # Filter the scores for the current consumer
        consumer_scores = scores[(scores['consumerid'] == consumer_id)]
        # Find the best match based on the joint score
        best_match = consumer_scores.loc[consumer_scores['jointScore'].idxmax()]
        best_matches.append(best_match)

    return pd.DataFrame(best_matches)


if __name__ == '__main__':
    if test:
        import doctest
        doctest.testmod()

    # Read the CSV data
    # filename = StringIO(data)
    # filename = 'dataset/consumption-3y.csv'
    # df = pd.read_csv(filename, sep=';', usecols=["Id", "Volumes"])

    # # Apply the function to the Volumes column
    # df['Volumes'] = df['Volumes'].apply(parse_volumes)

    # # Create a DataFrame from the 'Volumes' dictionaries
    # volumes_df = df['Volumes'].apply(pd.Series)

    # # Concatenate the original DataFrame with the new DataFrame of volumes
    # producer_df = pd.concat([df.drop(columns=['Volumes']), volumes_df], axis=1)
    # producer_df.to_csv('dataset/consumption-3y-processed.csv', sep=';', index=False)

    # # Print the final DataFrame
    # print(producer_df.info())
    # print(producer_df.head())

    consumer_df = pd.read_csv('dataset/consumption-1y-processed.csv', sep=';')
    producer_df = pd.read_csv('dataset/production-1y-processed.csv', sep=';')

    c_test_set, c_train_set, c_val_set = create_datasplit(consumer_df)
    p_test_set, p_train_set, p_val_set = create_datasplit(producer_df)
    _ = compute_and_store_simultaneity_scores(c_val_set, p_val_set, 'dataset/val_test_joint_scoring.csv')


