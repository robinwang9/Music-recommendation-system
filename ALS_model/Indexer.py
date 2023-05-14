from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

'''
Usage: spark-submit --deploy-mode client Indexer.py
Usage: spark-submit --deploy-mode cluster Indexer.py
Usage: spark-submit --deploy-mode cluster --num-executors 10 --executor-cores 4 Indexer.py
'''

def main(spark):
    df = spark.read.parquet("hdfs:/user/zz4140_nyu_edu/interactions_train_small_80.parquet")

    # Create a temporary view of the dataframe
    df.createOrReplaceTempView("interactions")

    # Drop the timestamp column
    df = df.drop("timestamp")

    # Count the number of times a user has listened to a song
    df_count = spark.sql("SELECT user_id, recording_msid, COUNT(*) AS count FROM interactions GROUP BY user_id, recording_msid")
    df_count = df_count.select(col("user_id"), col("recording_msid"), col("count").cast("integer"))

    # Use StringIndexer to convert string to numeric
    indexer_recording = StringIndexer(inputCol="recording_msid", outputCol="recording_msid_index", handleInvalid='skip')
    pipeline = Pipeline(stages=[indexer_recording])
    indexer = pipeline.fit(df_count)
    train_df = indexer.transform(df_count)

    train_df.write.mode("overwrite").parquet("indexed_train_small.parquet")
    # train_df.write.parquet("indexed_train_small.parquet")

    return train_df
    spark.stop()

if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName("Parquet Processing").config("spark.kryoserializer.buffer.max","512m").getOrCreate()

    # Get file_path for dataset to analyze
    #parquet_file_path = sys.argv[1]

    train_df = main(spark)