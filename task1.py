#!/usr/bin/env python
# coding: utf-8

# In[16]:


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
    interactions_train = spark.read.parquet("/scratch/work/courses/DSGA1004-2021/listenbrainz/interactions_train.parquet")
    interactions_test = spark.read.parquet("/scratch/work/courses/DSGA1004-2021/listenbrainz/interactions_test.parquet")

    # Subsample the data
    interactions_train = interactions_train.sample(False, subsample_rate, seed=42)

    # Split the training data into training and validation
    train, validation = interactions_train.randomSplit([0.6, 0.4], seed=42)

    # Apply tqdm progress bar to count rows in train and validation DataFrames
    train_row_count = train.rdd.mapPartitions(tqdm_count_rows).sum()
    validation_row_count = validation.rdd.mapPartitions(tqdm_count_rows).sum()

    print(f'Train row count: {train_row_count}')
    print(f'Validation row count: {validation_row_count}')

    return train, validation

if __name__ == '__main__':
    spark = SparkSession.builder.appName("partition_data")        .config("spark.executor.memory", "32g")        .config("spark.driver.memory", "32g")        .config("spark.sql.shuffle.partitions", "40")        .getOrCreate()

    subsample_rate = 0.5
    train, validation = partition_data(spark, subsample_rate)

    train.write.mode('overwrite').parquet("/scratch/jw5487/listenbrainz/interactions_train_partitioned.parquet")
    validation.write.mode('overwrite').parquet("/scratch/jw5487/listenbrainz/interactions_val_partitioned.parquet")


# In[17]:





# In[ ]:




