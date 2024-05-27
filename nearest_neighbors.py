from model_class import Model
from typing import Iterable
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
import evaluate
import sklearn as sk
import pandas as pd
import numpy as np
import preprocess

class NearestNeighbors(Model):
    def __init__(self):
        consumer_df = pd.read_csv('dataset/consumption-1y-processed.csv', sep=';')
        producer_df = pd.read_csv('dataset/production-1y-processed.csv', sep=';')

        c_test_set, c_train_set, c_val_set = preprocess.create_datasplit(consumer_df)
        p_test_set, p_train_set, p_val_set = preprocess.create_datasplit(producer_df)
        self.c_train_set = c_train_set
        self.p_train_set = p_train_set

        self.c_test_set = c_test_set
        self.p_test_set = p_test_set
        
        # Read csv file to numpy array excluding column ID
        consumption_data = c_train_set.to_numpy()[:,1:]
        production_data = p_train_set.to_numpy()[:,1:]

        self.scaler = StandardScaler()
        # Fit on both consumption and production data
        # Concatenate the two datasets
        global_data = np.concatenate((consumption_data, production_data), axis=0)
        self.scaler.fit(global_data)

        # Scale the consumption and production data
        consumption_scaled = self.scaler.transform(consumption_data)
        production_scaled = self.scaler.transform(production_data)

        self.consumption_tree = KDTree(consumption_scaled, leaf_size=4)              
        self.production_tree = KDTree(production_scaled, leaf_size=4)              

    
    def predict_consumers(self, producer_id: int) -> Iterable[tuple[int, float]]:
        """Producer id is in the test set."""
        # Get time series of producer
        producer_time_series = self.p_test_set[self.p_test_set['Id'] == producer_id].to_numpy()[:,1:]

        sample_scaled = self.scaler.transform(producer_time_series.reshape(1, -1))
        dist, ind = self.consumption_tree.query(sample_scaled, 10)
        # Return the index of the closest consumer
        # Take the specified rows from the ID column
        indices = self.c_train_set["Id"].iloc[ind[0]] 
        # Get the time series from the indices from the c_train_set
        time_series = self.c_train_set.iloc[ind[0]].to_numpy()[:,1:]
        

        
        scores = np.zeros_like(indices)

        return zip(indices, scores)


    def predict_producers(self, consumer_id: int) -> Iterable[tuple[int, float]]:
        # Get time series of producer

        consumer_time_series = self.c_test_set[self.c_test_set['Id'] == consumer_id].to_numpy()[:,1:]

        sample_scaled = self.scaler.transform(consumer_time_series.reshape(1, -1))
        dist, ind = self.production_tree.query(sample_scaled, 10)
        # Return the index of the closest consumer
        # Take the specified rows from the ID column
        indices = self.p_train_set["Id"].iloc[ind[0]] 
        scores = np.zeros_like(indices)

        return zip(indices, scores)
    

if __name__ == "__main__":
    model = NearestNeighbors()
    # print(model.predict_consumers(357))
    print(evaluate.evaluate_model(model))

    ...