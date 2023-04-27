import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *    
    
'''
This script is not completed yet. It does not work now.
还没写好，现在不能用!
'''

def data_split(spark, partial):
    '''
    Subsample the data to get a smaller dataset
        
    This function returns a dataframe corresponding to training, validation

    Parameters
    ----------
    spark : spark session object
    partial: float
        The subsampling rate
    '''
    # This loads the parquet file
    interactions = spark.read.parquet('hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet')

    interactions = interactions.sample(False, partial, seed= 42)
    interactions.createOrReplaceTempView('interactions')

    # Split the data into training, validation
    train, validation = interactions.randomSplit([0.8, 0.2], seed=42)
    train.createOrReplaceTempView('train')
    validation.createOrReplaceTempView('validation')

    return train, validation


if __name__ == '__main__':
    '''
    conf = SparkConf()
    conf.set("spark.executor.memory", "16G")
    conf.set("spark.driver.memory", '16G')
    conf.set("spark.executor.cores", "4")
    conf.set('spark.executor.instances','10')
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.default.parallelism", "40")
    conf.set("spark.sql.shuffle.partitions", "40")
    '''    

    spark = SparkSession.builder.appName("downsampling")\
	.config("spark.executor.memory", "32g")\
	.config("spark.driver.memory", "32g")\
	.config("spark.sql.shuffle.partitions", "40")\
	.getOrCreate()

    downsampling(spark)