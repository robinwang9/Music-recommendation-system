import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from tqdm import tqdm

def tqdm_count_rows(iterator):
    count = 0
    for row in tqdm(iterator, desc='Counting rows'):
        count += 1
    return [count]

def partition_data(spark, subsample_rate):
    # Read the data
    interactions_train = spark.read.parquet("hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet")
    interactions_test = spark.read.parquet("hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet")

    # Subsample the data
    interactions_train = interactions_train.sample(False, subsample_rate, seed=42)

    # Split the training data into training and validation
    train, validation = interactions_train.randomSplit([0.8, 0.2], seed=42)

    # Apply tqdm progress bar to count rows in train and validation DataFrames
    train_row_count = train.rdd.mapPartitions(tqdm_count_rows).sum()
    validation_row_count = validation.rdd.mapPartitions(tqdm_count_rows).sum()

    print(f'Train row count: {train_row_count}')
    print(f'Validation row count: {validation_row_count}')

    train.write.parquet("interactions_train.parquet")
    validation.write.parquet("interactions_val.parquet")
    return train, validation

if __name__ == '__main__':
    spark = SparkSession.builder.appName("partition_data")\
        .config("spark.executor.memory", "32g")\
        .config("spark.driver.memory", "32g")\
        .config("spark.sql.shuffle.partitions", "40")\
        .getOrCreate()

    subsample_rate = 0.5
    train, validation = partition_data(spark, subsample_rate)

    #train.write.mode('overwrite').parquet("hdfs:/user/zz4140/1004-project-2023/interactions_train.parquet")
    #validation.write.mode('overwrite').parquet("hdfs:/user/zz4140/1004-project-2023/interactions_val.parquet")