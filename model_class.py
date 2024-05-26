from abc import ABC, abstractmethod
from typing import Iterable
import pandas as pd
import numpy as np
from tslearn.utils import to_time_series_dataset
import preprocess


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
    def __init__(self, consumers, producers, consumer_clustering, consumer_scaler, producer_clustering, producer_scaler, df_representatives) -> None:
        self.consumers = consumers
        self.producers = producers
        self.consumer_clustering = consumer_clustering
        self.consumer_scaler = consumer_scaler
        self.producer_clustering = producer_clustering
        self.producer_scaler = producer_scaler
        self.df_representatives = df_representatives
        self.subsample_size = 10

    def _predict_cluster(self, df_timeseries: pd.DataFrame, consumer_clustering: bool):
        if consumer_clustering:
            cluster_model = self.consumer_clustering
            scaler = self.consumer_scaler
        else:
            cluster_model = self.producer_clustering
            scaler = self.producer_scaler

        data = df_timeseries.iloc[0, :].to_numpy()
        ts_data = to_time_series_dataset(data)
        ts_scaled = scaler.fit_transform(ts_data)
        cluster = cluster_model.predict(ts_scaled)
        return cluster


    def _compare_with_clusters(self, df_timeseries: pd.DataFrame, consumer_to_producer: bool) -> Iterable[tuple[int, int, float]]:
        if consumer_to_producer:
            ids_representatives = self.df_representatives["representative_ids_production1y"]
            representatives = self.df_representatives["representative_ids_production1y"]
        else:
            ids_representatives = self.df_representatives["representative_ids_consumption1y"]
            representatives = self.df_representatives["representatives_consumption1y"]

        scores = []
        for idx, representative in enumerate(representatives):
            score = preprocess.compute_joint_score(df_timeseries.iloc[0,:].to_numpy(), representative.iloc[idx, :].to_numpy())
            # id of cluster, id of representative, score
            scores.append((idx, ids_representatives.iloc[idx], score))
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores


    def _compare_within_cluster(self, user_id: int, cluster: int, num_samples: int, consumer_to_producer: bool) -> Iterable[tuple[int, int, float]]:
        if consumer_to_producer:
            df_timeseries = self.consumers[self.consumers.index == user_id]
            df_compare_to = self.producers
        else:
            df_timeseries = self.producers[self.producers.index == user_id]
            df_compare_to = self.consumers
        # Select rows that belong to the cluster
        df_cluster = df_compare_to[df_compare_to["cluster"] == cluster]
        users_to_compare_to = df_cluster.sample(num_samples, replace=False)
        scores = []
        for idx, user in users_to_compare_to.iterrows():
            score = preprocess.compute_joint_score(df_timeseries.iloc[0,:].to_numpy(), user.to_numpy())
            # id of cluster, id of representative, score
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def predict_consumers(self, producer_id: int) -> Iterable[tuple[int, float]]:
        cluster_scores = self._compare_with_clusters(self.producers[self.producers.index == producer_id], consumer_to_producer=False)
        cluster = cluster_scores[0][0]
        scores = self._compare_within_cluster(producer_id, cluster, self.subsample_size, consumer_to_producer=False)
        return scores


    def predict_producers(self, consumer_id: int) -> Iterable[tuple[int, float]]:
        cluster_scores = self._compare_with_clusters(self.producers[self.producers.index == consumer_id], consumer_to_producer=True)
        cluster = cluster_scores[0][0]
        scores = self._compare_within_cluster(consumer_id, cluster, self.subsample_size, consumer_to_producer=True)
        return scores