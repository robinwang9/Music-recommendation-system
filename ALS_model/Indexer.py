from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

'''
Usage: spark-submit --deploy-mode client Indexer.py
Usage: spark-submit --deploy-mode cluster Indexer.py
Usage: spark-submit --deploy-mode cluster --num-executors 10 --executor-cores 4 Indexer.py
'''

def main(spark):
    df = spark.read.parquet("hdfs:/user/zz4140_nyu_edu/interactions_train_small_80.parquet")

    # Use StringIndexer to convert string to numeric
    #indexer_recording = StringIndexer(inputCol="recording_msid", outputCol="recording_msid_index")
    #df = indexer_recording.fit(df).transform(df)

    # Drop the timestamp column
    df = df.drop("timestamp")

    # Count the number of times a user has listened to a song
    #df_count = df.groupBy("user_id", "recording_msid_index").count()
    #df_count = df_count.withColumnRenamed("count", "count_combination")

    df.write.parquet("indexed_train_small.parquet")

    return df
    spark.stop()

if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName("Parquet Processing").config("spark.kryoserializer.buffer.max","512m").getOrCreate()

    # Get file_path for dataset to analyze
    #parquet_file_path = sys.argv[1]

    df = main(spark)