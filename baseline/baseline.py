#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Script to evaluate a baseline popularity model
Usage:
    $ spark-submit --driver-memory 8g --executor-memory 8g baseline.py hdfs:/user/zz4140_nyu_edu/items_popular.parquet hdfs:/user/zz4140_nyu_edu/interactions_val.parquet
    or spark-submit baseline.py hdfs:/user/zz4140_nyu_edu/items_popular.parquet hdfs:/user/zz4140_nyu_edu/interactions_val.parquet
Returns:
    Precision at 15
    MAP
    NDCG
'''

# Import packages
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
#import time

def main(spark, input_file_path, input_val_file_path):
    '''
    Parameters
    ----------
    spark : SparkSession object
    input_file_path: top 500 tracks hdfs:/user/zz4140_nyu_edu/items_popular.parquet
    input_val_file_path: sorted validation data to use hdfs:/user/zz4140_nyu_edu/interactions_val.parquet
    '''        
    # Loads the parquet files
    track_pop = spark.read.parquet(input_file_path)
    track_val = spark.read.parquet(input_val_file_path)
    
    # Create a hash column as the ID column
    track_pop = track_pop.withColumn("track_hashId", func.hash("recording_msid"))
    track_val = track_val.withColumn("user_hashId", func.hash("user_id"))
    track_val = track_val.withColumn("track_hashId", func.hash("recording_msid"))
    
    # Get the users from the val file
    users = track_val.select(track_val.user_hashId).distinct() 
    
    # Get cross product of users and popular tracks
    user_pop_track = users.crossJoin(track_pop)
    
    # Aggregate to get popular tracks into a list
    user_pop_agg = user_pop_track.groupby("user_hashId").agg(func.collect_list("track_hashId"))
    user_pop_agg = user_pop_agg.withColumnRenamed("collect_list(track_hashId)", "pop_track")

    
    # Collapse validation file in the same manner
    track_val_agg = track_val.groupby("user_hashId").agg(func.collect_list("track_hashId"))
    track_val_agg = track_val_agg.withColumnRenamed("collect_list(track_hashId)", "truth")
   
   
    # EVALUATION
    predictionAndLabels = user_pop_agg.join(track_val_agg, ["user_hashId"])
    predictionAndLabels = predictionAndLabels.select("pop_track", "truth")
    predictionAndLabels_rdd = predictionAndLabels.rdd
    
    metrics = RankingMetrics(predictionAndLabels_rdd)

    
    # Call ranking evaluation metrics
    pr_at = 15
    print('Precision at'+str(pr_at)+':')
    print(metrics.precisionAt(pr_at))
    print('MAP:')
    print(metrics.meanAveragePrecision)
    print('NDCG:')
    print(metrics.ndcgAt(10))

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('BaselinePopularity').getOrCreate()

    # Get file_path for dataset to analyze
    input_file_path = sys.argv[1]
    input_val_file_path = sys.argv[2]

    main(spark, input_file_path, input_val_file_path)