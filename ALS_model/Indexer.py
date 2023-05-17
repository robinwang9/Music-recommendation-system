from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql.functions import row_number
from pyspark.sql.window import Window

'''
Usage: spark-submit --deploy-mode client Indexer.py
Usage: spark-submit --deploy-mode cluster Indexer.py
Usage: spark-submit --deploy-mode cluster --num-executors 10 --executor-cores 4 Indexer.py
'''

def main(spark):
    #df = spark.read.parquet("hdfs:/user/zz4140_nyu_edu/interactions_train_small_80.parquet")
    #df = spark.read.parquet("hdfs:/user/zz4140_nyu_edu/interactions_val_small_20.parquet")
    df = spark.read.parquet("hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet")

    # Create a temporary view of the dataframe
    df.createOrReplaceTempView("interactions")

    # Drop the timestamp column
    df = df.drop("timestamp")

    # Count the number of times a user has listened to a song
    df_count = spark.sql("SELECT user_id, recording_msid, COUNT(*) AS count FROM interactions GROUP BY user_id, recording_msid")
    df_count = df_count.select(col("user_id"), col("recording_msid"), col("count").cast("integer"))

    # Extract the distinct recording_msid from the original training set
    df_original_train = spark.read.parquet("hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet")
    distinct_df = df_original_train.select("recording_msid").distinct()

    # Add a column of recording_idx
    distinct_df = distinct_df.withColumn("recording_idx", row_number().over(Window.orderBy("recording_msid")))

    # Join the distinct recording_msid with the count dataframe
    merged_df = df_count.join(distinct_df, on="recording_msid", how="inner").drop("recording_msid")


    merged_df.repartition(5000, 'recording_idx').write.mode("overwrite").parquet("indexed_test.parquet")
    #merged_df.repartition(5000, 'recording_idx').write.mode("overwrite").parquet("indexed_val_small.parquet")
    # train_df.write.parquet("indexed_train_small.parquet")

    return merged_df
    spark.stop()

if __name__ == "__main__":
    # Create the spark session object
    spark = (SparkSession.builder.appName("Parquet Processing").config("spark.driver.memory", '16G').config('spark.executor.memory','20g').config('spark.dynamicAllocation.enabled', True).config('spark.dynamicAllocation.minExecutors',3).getOrCreate())

    merged_df = main(spark)