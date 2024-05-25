import pandas as pd
import json
from io import StringIO
import sklearn.model_selection

# Sample data
data = """Id;End;Start;Volumes
2;2026-01-01T00:00:00Z;2025-01-01T00:00:00Z;[{'Key': '2025-01-01:00', 'Value': 43.48}, {'Key': '2025-01-01:01', 'Value': 43.659}, {'Key': '2025-01-01:02', 'Value': 41.7}]
"""
# Read the CSV data
# filename = StringIO(data)
filename = 'dataset/consumption-1y.csv'
df = pd.read_csv(filename, sep=';', usecols=["Id", "Volumes"])

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


if __name__ == '__main__':
    # Apply the function to the Volumes column
    df['Volumes'] = df['Volumes'].apply(parse_volumes)

    # Create a DataFrame from the 'Volumes' dictionaries
    volumes_df = df['Volumes'].apply(pd.Series)

    # Concatenate the original DataFrame with the new DataFrame of volumes
    result_df = pd.concat([df.drop(columns=['Volumes']), volumes_df], axis=1)

    # Print the final DataFrame
    print(result_df.info())
    print(result_df.head())
