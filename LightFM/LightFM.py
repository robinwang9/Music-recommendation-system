import numpy as np
import pandas as pd
import time
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import lightfm
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import matplotlib.pyplot as plt
from pyspark.sql.functions import col

print('Final Project Extension LightFM on Full dataset')

# Read datasets
df_train = spark.read.parquet('/scratch/jw5487/data_final/train_data.parquet').sample(fraction=0.5, seed=42)
df_val = spark.read.parquet('/scratch/jw5487/data_final/validation_data.parquet').sample(fraction=0.5, seed=42)
df_test = spark.read.parquet('scratch/jw5487/data_final/test.parquet').sample(fraction=0.5, seed=42)

# Convert to pandas
ratings_full_train = df_train.select("user_id", "recording_msid", "timestamp", "row_number").toPandas()
ratings_full_val = df_val.select("user_id", "recording_msid", "timestamp", "row_number", "train_split").toPandas()
ratings_full_test = df_test.select("user_id", "recording_msid", "timestamp").toPandas()

# Rename columns to match the original code
ratings_full_train.columns = ['userId', 'movieId', 'rating', 'timestamp']
ratings_full_val.columns = ['userId', 'movieId', 'rating', 'timestamp', "median_timestamp"]
ratings_full_test.columns = ['userId', 'movieId', 'rating', 'timestamp']

print('Dropping irrelavant columns')
# drop median_timestamp & timestamp columns
ratings_full_val = ratings_full_val.drop(columns=['median_timestamp'])
ratings_full_test = ratings_full_test.drop(columns=['median_timestamp'])
ratings_full_train = ratings_full_train.drop(columns=['timestamp'])
ratings_full_val = ratings_full_val.drop(columns=['timestamp'])
ratings_full_test = ratings_full_test.drop(columns=['timestamp'])

print('Reassigning recording_msid to avoid dimension error')
# reassign recording_msid to avoid dimension error when fitting LightFM model
total_item_user = pd.concat([ratings_full_train, ratings_full_test, ratings_full_val]).drop_duplicates()
total_item_user = total_item_user.sort_values(['movieId'])
total_item_user['new_recording_msid'] = (total_item_user.groupby(['movieId'], sort=False).ngroup()+1)

print('Appending new_recording_msid to existing train, test, and validation sets')
# append new_recording_msid to existing train, test, and validation test sets
ratings_full_train = ratings_full_train.merge(total_item_user, on=['movieId','userId','rating'], how="left")
ratings_full_val = ratings_full_val.merge(total_item_user, on=['movieId','userId','rating'], how="left")
ratings_full_test = ratings_full_test.merge(total_item_user, on=['movieId','userId','rating'], how="left")

print('Dropping original recording_msid columns')
# drop original recording_msid column
ratings_full_train = ratings_full_train.drop(columns=['movieId'])
ratings_full_val = ratings_full_val.drop(columns=['movieId'])
ratings_full_test = ratings_full_test.drop(columns=['movieId'])

print("Adjusting dataset dimensions to avoid unmatched dimension error")
# adjust dataset dimensions to avoid error of unmatched dimensions
data = Dataset()
data.fit(users = np.unique(total_item_user["userId"]), items = np.unique(total_item_user["new_recording_msid"]))

print("building interactions")
# build interactions
interactions_train, weights_train = data.build_interactions([(ratings_full_train['userId'][i], 
                                                              ratings_full_train['new_recording_msid'][i],
                                                              ratings_full_train['rating'][i]) for i in range(ratings_full_train.shape[0])])
interactions_val, weights_val = data.build_interactions([(ratings_full_val['userId'][i],
                                                          ratings_full_val['new_recording_msid'][i], 
                                                          ratings_full_val['rating'][i]) for i in range(ratings_full_val.shape[0])])

# Define function to train and evaluate a model with given parameters
def train_evaluate_model(loss, no_components, user_alpha, interactions_train, weights_train, interactions_val):
    model = LightFM(loss=loss, no_components=no_components, user_alpha=user_alpha)
    model.fit(interactions=interactions_train, sample_weight=weights_train, epochs=1, verbose=False)
    val_precision = precision_at_k(model, interactions_val, k=100).mean()
    return val_precision

# Parameter search
ranks = [10, 20, 30, 40, 50]
regularizations = [0.0005, 0.005, 0.01, 0.1]

best_params = None
best_precision = -np.inf
for rank in ranks:
    for reg in regularizations:
        print(f"Training WARP model with rank: {rank}, regularization: {reg}")
        precision = train_evaluate_model('warp', rank, reg, interactions_train, weights_train, interactions_val)
        print(f"Precision at k=100: {precision}")
        if precision > best_precision:
            best_precision = precision
            best_params = (rank, reg)

best_rank, best_reg = best_params
print(f"Best parameters: rank={best_rank}, regularization={best_reg}")

# Train the best model
print("Training the best WARP model with the selected parameters")
start = time.time()
best_model = LightFM(loss='warp', no_components=best_rank, user_alpha=best_reg)
best_model.fit(interactions=interactions_train, sample_weight=weights_train, epochs=1, verbose=False)
val_precision = precision_at_k(best_model, interactions_val, k=100).mean()
end = time.time()

print("LightFM WARP model on full dataset")
print("Precision at k is:", val_precision)
print("Time spent is:", end - start)
