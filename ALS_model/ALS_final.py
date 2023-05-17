import sys
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline, PipelineModel
import random
import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import itertools
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col, expr

'''
Usage:
$ spark-submit --driver-memory=8g --executor-memory=8g --conf "spark.blacklist.enabled=false" ALS_final.py hdfs:/user/zz4140_nyu_edu/indexed_train_small.parquet hdfs:/user/zz4140_nyu_edu/indexed_test.parquet
$ spark-submit --deploy-mode client ALS_final.py hdfs:/user/zz4140_nyu_edu/indexed_train_small.parquet hdfs:/user/zz4140_nyu_edu/indexed_test.parquet
'''

def main(spark, train_path, val_path):
    '''
    '''
    train = spark.read.parquet(train_path)
    val = spark.read.parquet(val_path)

    user_id = val.select('user_id').distinct()
    true_tracks = val.select('user_id', 'recording_idx').groupBy('user_id').agg(expr('collect_list(recording_idx) as tracks'))

    als = ALS(maxIter=10, userCol ='user_id', itemCol = 'recording_idx', implicitPrefs = True, \
        nonnegative=True, ratingCol = 'count', rank = 50, regParam = 0.05, alpha = 1)
    model = als.fit(train)

    pred_tracks = model.recommendForUserSubset(user_id,100)
    pred_tracks = pred_tracks.select("user_id", col("recommendations.recording_idx").alias("tracks")).sort('user_id')

    tracks_rdd = pred_tracks.join(F.broadcast(true_tracks), 'user_id', 'inner') \
                .rdd.map(lambda row: (row[1], row[2]))
    metrics = RankingMetrics(tracks_rdd)
    map = metrics.meanAveragePrecision
    prec = metrics.precisionAt(100)
    ndcg = metrics.ndcgAt(100)
    print('meanAveragePrecision: ', map, 'precisionAt: ', prec, 'ndcg: ', ndcg )

    preds = model.transform(val)
    reg_evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",predictionCol="prediction")
    rmse = reg_evaluator.evaluate(preds)
    print('rmse: ', rmse)

    print('Saving latent factor')
    if_df = model.itemFactors
    uf_df = model.userFactors

    #if_df.repartition(1).write,format("parquet").save("itemFactors_50_0.05_1.parquet")
    #uf_df.repartition(1).write,format("parquet").save("userFactors_50_0.05_1.parquet")


# Only enter this block if we're in main
if __name__ == "__main__":

    spark = (SparkSession.builder.appName('first_step').config("spark.driver.memory", '16G').config('spark.executor.memory','20g').config('spark.dynamicAllocation.enabled', True).config('spark.dynamicAllocation.minExecutors',3).getOrCreate())
    # sc = SparkContext.getOrCreate()

    # Get file_path for dataset to analyze
    train_path = sys.argv[1]
    val_path = sys.argv[2]

    main(spark, train_path, val_path)