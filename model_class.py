from abc import ABC, abstractmethod
from typing import Iterable
import pandas as pd
import numpy as np
from tslearn.utils import to_time_series_dataset
import preprocess
import pickle
import evaluate

class Model(ABC):
    """Model that can take in 1 producer and give id of 10 consumers that are most likely to consume the energy produced by the producer. And take in 1 consumer and give id of 10 producers that are most likely to produce the energy consumed by the consumer."""

    @abstractmethod
    def predict_consumers(self, producer_id: int) -> Iterable[tuple[int, float]]:
        """Predict the 10 consumers that are most likely to consume the energy produced by the producer."""
        # Throw not implemented error
        raise NotImplementedError

    @abstractmethod
    def predict_producers(self, consumer_id: int) -> Iterable[tuple[int, float]]:
        """Predict the 10 producers that are most likely to produce the energy consumed by the consumer."""
        # Throw not implemented error
        raise NotImplementedError


class TimeseriesClusteringModel(Model):
    """Model based on the timeseries cluseting to predict the 10 most likely consumers and producers for a given producer and consumer respectively."""
    def __init__(self, df_consumers, df_producers, df_representatives, sumsample_size=10) -> None:
        self.df_consumers = df_consumers
        self.df_producers = df_producers
        self.df_representatives = df_representatives
        self.subsample_size = sumsample_size


    def _compare_with_clusters(self, df_timeseries: pd.DataFrame, consumer_to_producer: bool) -> Iterable[tuple[int, int, float]]:
        if consumer_to_producer:
            df_compare_to = self.df_producers
            ids_representatives = self.df_representatives["representative_ids_production1y"]
            representatives = self.df_representatives["representatives_production1y"]
        else:
            df_compare_to = self.df_consumers
            ids_representatives = self.df_representatives["representative_ids_consumption1y"]
            representatives = self.df_representatives["representatives_consumption1y"]

        scores = []
        for idx, representative in enumerate(representatives):
            # Acces timeseries of the representative
            score = preprocess.compute_joint_score(df_timeseries.iloc[0,1:-1].to_numpy(),
                                                   representative)
            # id of cluster, id of representative, score
            scores.append((idx, ids_representatives.iloc[idx], score))
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores


    def _compare_within_cluster(self, user_id: int, cluster: int, num_samples: int, consumer_to_producer: bool) -> Iterable[tuple[int, int, float]]:
        if consumer_to_producer:
            df_timeseries = self.df_consumers[self.df_consumers["Id"] == user_id]
            df_compare_to = self.df_producers
        else:
            df_timeseries = self.df_producers[self.df_producers["Id"] == user_id]
            df_compare_to = self.df_consumers
        # Select rows that belong to the cluster
        df_cluster = df_compare_to[df_compare_to["Cluster"] == cluster]
        if len(df_cluster) < num_samples:
            users_to_compare_to = df_cluster
        else:
            users_to_compare_to = df_cluster.sample(num_samples, replace=False)
        scores = []
        for idx, user in users_to_compare_to.iterrows():
            score = preprocess.compute_joint_score(df_timeseries.iloc[0,1:-1].to_numpy(), user.iloc[1:-1].to_numpy())
            # id of cluster, id of representative, score
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def predict_consumers(self, producer_id: int) -> Iterable[tuple[int, float]]:
        cluster_scores = self._compare_with_clusters(self.df_producers[self.df_producers["Id"] == producer_id], consumer_to_producer=False)
        cluster = cluster_scores[0][0]
        scores = self._compare_within_cluster(producer_id, cluster, self.subsample_size, consumer_to_producer=False)
        return scores


    def predict_producers(self, consumer_id: int) -> Iterable[tuple[int, float]]:
        cluster_scores = self._compare_with_clusters(self.df_consumers[self.df_consumers["Id"] == consumer_id], consumer_to_producer=True)
        cluster = cluster_scores[0][0]
        scores = self._compare_within_cluster(consumer_id, cluster, self.subsample_size, consumer_to_producer=True)
        return scores

if __name__ == "__main__":
    df_consumers = pd.read_csv('clustered_consumers.csv')
    df_producers = pd.read_csv('clustered_producers.csv')
    with open("models/representatives_df.pkl", 'rb') as f:
        df_representatives = pickle.load(f)

    model = TimeseriesClusteringModel(df_consumers, df_producers, df_representatives)
    print(evaluate.evaluate_model(model))
