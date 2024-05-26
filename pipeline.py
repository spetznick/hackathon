import pandas as pd
import numpy as np
import preprocess

if __name__ == "__main__":
    # Load the data
    df_consumer = pd.read_csv("dataset/consumption-1y-processed.csv")
    df_producer = pd.read_csv("dataset/production-1y-processed.csv")
    val_test_joint_scoring = pd.read_csv("dataset/val_test_joint_scoring.csv")

    # Load the scaler
    # Load the model

    # Load the representatives of the clusters
    # Run pipeline on validation set

    # Evaluate the model
    # Compare with the precomputed simultaneity scores