#%%
import pandas as pd
import numpy as np
import pickle
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

def load_model_scaler(model_path, scaler_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_cluster(model, scaler, data):
    ts_data = to_time_series_dataset([data])
    ts_scaled = scaler.transform(ts_data)
    cluster = model.predict(ts_scaled)
    return cluster[0]

def compute_simultaneity(user1_timeries, user2_timeries):
    sum_user1 = np.sum(user1_timeries)
    sum_of_max = np.sum(np.maximum(user1_timeries - user2_timeries, 0))
    simultaneity_score = 1 - sum_of_max / (sum_user1 + 0.000001)
    return simultaneity_score

def compute_and_store_simultaneity_scores(df_consumers, df_producers, filename=None):
    entries = []
    for consumer_id, consumer in df_consumers.iterrows():
        for producer_id, producer in df_producers.iterrows():
            if producer_id == consumer_id:  # skip calculating for same ID
                continue
            score_c2p = compute_simultaneity(consumer[:-1], producer[:-1])  # Excluding the 'Cluster' column
            score_p2c = compute_simultaneity(producer[:-1], consumer[:-1])
            joint_score = score_c2p + score_p2c - abs(score_c2p - score_p2c)
            entries.append({
                'consumerid': consumer_id, 'producerid': producer_id,
                'scoreC2P': score_c2p, 'scoreP2C': score_p2c, 'jointScore': joint_score
            })

    results_df = pd.DataFrame(entries)
    results_df.sort_values(by='jointScore', ascending=False, inplace=True)
    if filename:
        results_df.to_csv(filename, index=False)
    return results_df

# Load data
consumer_data = pd.read_csv('dataset/consumption-1y-processed.csv', sep=';', index_col=0)
producer_data = pd.read_csv('dataset/production-1y-processed.csv', sep=';', index_col=0)

# Load models and scalers
consumer_model, consumer_scaler = load_model_scaler('models/model_consumption1y.pkl', 'models/scaler_consumption1y.pkl')
producer_model, producer_scaler = load_model_scaler('models/model_production1y.pkl', 'models/scaler_production1y.pkl')

# Predict clusters
consumer_data['Cluster'] = consumer_data.apply(lambda x: predict_cluster(consumer_model, consumer_scaler, x), axis=1)
producer_data['Cluster'] = producer_data.apply(lambda x: predict_cluster(producer_model, producer_scaler, x), axis=1)

#save a clustered_consumers.csv
consumer_data.to_csv('dataset/clustered_consumers.csv', index=False)
producer_data.to_csv('dataset/clustered_producers.csv', index=False)

# Compute simultaneity scores
results_df = compute_and_store_simultaneity_scores(consumer_data, producer_data, 'simultaneity_scores.csv')
