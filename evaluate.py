import preprocess
import pandas as pd
from typing import Iterable
from dataclasses import dataclass

def precompute_evaluation_data(file_suffix: str):
    """Precompute evaluation data for the test set. file_suffix is what comes after 'consumption' or 'production' but before the file extension"""
    # Load the test set
    df_consumer = pd.read_csv(f"dataset/consumption{file_suffix}.csv", sep=';')
    df_producer = pd.read_csv(f"dataset/production{file_suffix}.csv", sep=';')

    c_test_set, c_train_set, c_val_set = preprocess.create_datasplit(df_consumer)
    p_test_set, p_train_set, p_val_set = preprocess.create_datasplit(df_producer)

    preprocess.compute_and_store_simultaneity_scores(c_test_set, p_train_set, f'dataset/consumer-evaluation{file_suffix}.csv')
    preprocess.compute_and_store_simultaneity_scores(c_train_set, p_test_set, f'dataset/producer-evaluation{file_suffix}.csv')


@dataclass
class EvaluationResult:
    """Class to store the results of the evaluation"""
    avg_correct_consumer_predictions: float # Value between 0 and 10
    average_joint_score_consumers: float # Value between 0 and 1
    avg_correct_producer_predictions: float # Value between 0 and 10
    average_joint_score_producers: float


def evaluate_model(model):
    file_suffix = "-1y-processed"
    # Load the test set
    consumer_evaluation = pd.read_csv(f"dataset/consumer-evaluation{file_suffix}.csv", sep=',')
    producer_evaluation = pd.read_csv(f"dataset/producer-evaluation{file_suffix}.csv", sep=',')

    consumer_indices = consumer_evaluation['consumerid'].unique()

    avg_correct_predictions = 0
    average_joint_score = 0
    # For each unique consumer index in the consumer evaluation set predict the 10 most likely producers
    for consumer_id in consumer_indices:
        # Get the 10 most likely producers
        predictions = model.predict_producers(consumer_id)
        # Get the actual 10 most likely producers
        producer_ranking = consumer_evaluation[consumer_evaluation['consumerid'] == consumer_id]
        actual_top_10 = producer_ranking.nlargest(10, 'jointScore')["producerid"].to_numpy()

        # Calculate the number of correct predictions
        for producer_id, joint_score in predictions:
            average_joint_score += joint_score
            if producer_id in actual_top_10:
                avg_correct_predictions += 1
    avg_correct_predictions /= len(consumer_indices)
    average_joint_score /= 10 * len(consumer_indices)

    # Do the same for the producer evaluation set
    producer_indices = producer_evaluation['producerid'].unique()
    avg_correct_consumer_prodictions = 0
    average_joint_score_producer = 0

    for producer_id in producer_indices:
        # Get the 10 most likely consumers
        predictions = model.predict_consumers(producer_id)
        # Get the actual 10 most likely consumers
        consumer_ranking = producer_evaluation[producer_evaluation['producerid'] == producer_id]
        actual_top_10 = consumer_ranking.nlargest(10, 'jointScore')["consumerid"].to_numpy()

        # Calculate the number of correct predictions
        for consumer_id, joint_score in predictions:
            average_joint_score_producer += joint_score
            if consumer_id in actual_top_10:
                avg_correct_consumer_prodictions += 1
    avg_correct_consumer_prodictions /= len(producer_indices)
    average_joint_score_producer /= 10 * len(producer_indices)


    return EvaluationResult(avg_correct_consumer_prodictions, average_joint_score_producer, avg_correct_predictions, average_joint_score)



class DummyModel:

    def predict_consumers(self, producer_id: int) -> Iterable[tuple[int, float]]:
        # return [(1, 0.5), (2, 0.4), (3, 0.3), (3, 0.2), (2, 0.1), (2, 0.1), (2, 0.1), (2, 0.1), (2, 0.1), (2, 0.1)]
        return [(1, 0.5), (2, 0.4), (3, 0.3), (4, 0.2), (5, 0.1), (6, 0.1), (7, 0.1), (8, 0.1), (9, 0.1), (10, 0.1)]

    def predict_producers(self, consumer_id: int) -> Iterable[tuple[int, float]]:
        return [(1, 0.5), (2, 0.4), (3, 0.3), (4, 0.2), (5, 0.1), (6, 0.1), (7, 0.1), (8, 0.1), (9, 0.1), (10, 0.1)]


if __name__ == "__main__":
    precompute_evaluation_data("-1y-processed")
    print(evaluate_model(DummyModel()))