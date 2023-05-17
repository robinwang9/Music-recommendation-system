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
$ spark-submit --deploy-mode client single_params_train.py hdfs:/user/zz4140_nyu_edu/indexed_train_small.parquet hdfs:/user/zz4140_nyu_edu/indexed_val_small.parquet
$ spark-submit --deploy-mode cluster single_params_train.py hdfs:/user/zz4140_nyu_edu/indexed_train_small.parquet hdfs:/user/zz4140_nyu_edu/indexed_val_small.parquetls
$ spark-submit --deploy-mode cluster --num-executors 10 --executor-cores 4 single_params_train.py hdfs:/user/zz4140_nyu_edu/indexed_train_small.parquet hdfs:/user/zz4140_nyu_edu/indexed_val_small.parquet
'''

def main(spark, train_path, val_path):
    # Load train, val data
    train = spark.read.parquet(train_path)
    val = spark.read.parquet(val_path)

    # Use StringIndexer to convert string to numeric
    # indexer_recording = StringIndexer(inputCol="recording_msid", outputCol="recording_idx", handleInvalid='skip')
    # pipeline = Pipeline(stages=[indexer_recording])
    # indexer_train = pipeline.fit(train)
    # indexer_val = pipeline.fit(val)
    # train_df = indexer_train.transform(train)
    # val_df = indexer_val.transform(val)

    # user_index = PipelineModel.load(indexer_model)
    # val = user_index.transform(val)
    # val = val.select('user_idx','recording_idx','count')

    user_id = val.select('user_id').distinct()
    true_tracks = val.select('user_id', 'recording_idx').orderBy('user_id',"count",ascending=False).groupBy('user_id').agg(expr('collect_list(recording_idx) as tracks'))

    rank_val =  10 #default is 10
    reg_val =  0.05  #default is 1
    alpha_val = 1 #default is 1

    # maps=[]
    # precs=[]
    # ndcgs=[]
    # rmses=[]

    #for i in rank_val: #change to reg or alpha #then set the rest to default
    als = ALS(maxIter=10, userCol ='user_id', itemCol = 'recording_idx', implicitPrefs = True, nonnegative=True, ratingCol = 'count', rank = rank_val, regParam = reg_val, alpha = alpha_val, numUserBlocks = 50, numItemBlocks = 50, seed=123)
    model = als.fit(train)

    pred_tracks = model.recommendForUserSubset(user_id,100)
    pred_tracks = pred_tracks.select("user_id", col("recommendations.recording_idx").alias("tracks")).sort('user_id')

    tracks_rdd = pred_tracks.join(F.broadcast(true_tracks), 'user_id', 'inner').rdd.map(lambda row: (row[1], row[2]))
    metrics = RankingMetrics(tracks_rdd)
    map = metrics.meanAveragePrecision
    prec = metrics.precisionAt(100)
    ndcg = metrics.ndcgAt(100)

    # maps.append(map)
    # precs.append(prec)
    # ndcgs.append(ndcg)
    # print('rank: ', rank_val, 'regularization param: ', reg_val, 'alpha: ', alpha_val )
    # print('meanAveragePrecision: ', map, 'precisionAt: ', prec, 'ndcg: ', ndcg )

    #preds = model.transform(val)
    #reg_evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",predictionCol="prediction")
    #rmse = reg_evaluator.evaluate(preds)

    # rmses.append(rmse)
    #print('rmse: ', rmse)
    #print(f"finished training ALS model with rank{rank_val} and reg{reg_val} and alpha {alpha_val}")
    #model.save(f"hdfs:/user/zn2041/ALS_model_rank{rank_val}_reg{reg_val}_alpha{alpha_val}")

    # print('ranks: ', rank_val)
    # print('maps: ', maps)
    # print('precs: ', precs)
    # print('ncdgs: ', ndcgs)
    # print('rmses: ', rmses)



# Only enter this block if we're in main
if __name__ == "__main__":
    #conf = SparkConf()
    #conf.set("spark.executor.memory", "16G")
    #conf.set("spark.driver.memory", '16G')
    #conf.set("spark.executor.cores", "4")
    #conf.set('spark.executor.instances','10')
    #conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    #conf.set("spark.default.parallelism", "40")
    #conf.set("spark.sql.shuffle.partitions", "40")
    #spark = SparkSession.builder.config(conf=conf).appName('first_train').getOrCreate()

    #spark = SparkSession.builder.appName('first_step').config("spark.driver.memory", '16G').config('spark.executor.memory','20g').config('spark.dynamicAllocation.enabled', True).config('spark.dynamicAllocation.minExecutors',3).getOrCreate()
    spark = SparkSession.builder.appName('first_step').getOrCreate()
    # Get file_path for dataset to analyze
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    #indexer_model = sys.argv[3]

    main(spark, train_path, val_path)