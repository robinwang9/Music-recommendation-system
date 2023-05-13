'''
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
    interactions_test = spark.read.parquet("hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet")

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
'''
'''
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import count, row_number, rand
from pyspark.sql.functions import round

def main():
    df = spark.read.parquet("hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet")

    counts = df.groupBy("user_id").count()

    valid_user_ids = counts.filter(counts["count"] > 5).select("user_id")

    cleaned_df = df.join(valid_user_ids, "user_id")

    user_counts = cleaned_df.groupBy("user_id").count()

    user_counts = user_counts.withColumn("rounded_count", round(user_counts["count"]))

    train_counts = (user_counts["rounded_count"] * 0.8).cast("integer")
    val_counts = user_counts["rounded_count"] - train_counts

    train_df = cleaned_df.sampleBy("user_id", fractions={row["user_id"]: train_counts[row["user_id"]]/row["count"] for row in user_counts.collect()}, seed=42)
    val_df = cleaned_df.subtract(train_df)

    train_df.write.parquet("interactions_train_small_80.parquet")
    val_df.write.parquet("interactions_val_small_20.parquet")

    spark.stop()

'''
def main():
    # Read the input parquet file
    interactions_train = spark.read.parquet("hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet")

    # Count interactions for each user
    user_interactions_count = interactions_train.groupBy('user_id').agg(count('recording_msid').alias('interactions_count'))

    # Filter users with at least 5 interactions
    filtered_users = user_interactions_count.filter(user_interactions_count.interactions_count >= 5)

    # Join the filtered_users dataframe with the interactions_train dataframe
    filtered_interactions = interactions_train.join(filtered_users.select('user_id'), on='user_id', how='inner')

    # Add row_number for each user's interactions, partitioned by user_id and ordered by timestamp
    window = Window.partitionBy('user_id').orderBy('timestamp')
    filtered_interactions = filtered_interactions.withColumn('row_number', row_number().over(window))

    # Calculate the 80% split for each user
    split_ratio = 0.8
    filtered_users = filtered_users.withColumn('train_split', (filtered_users.interactions_count * split_ratio).cast('int'))

    # Join the filtered_interactions dataframe with the train_split value for each user
    filtered_interactions = filtered_interactions.join(filtered_users.select('user_id', 'train_split'), on='user_id', how='inner')

    # Split the data into training and validation sets
    train_data = filtered_interactions.filter(filtered_interactions.row_number <= filtered_interactions.train_split)
    validation_data = filtered_interactions.filter(filtered_interactions.row_number > filtered_interactions.train_split)

    # Save the training and validation data as parquet files
    train_data.write.parquet("interactions_train_small_80.parquet")
    validation_data.write.parquet("interactions_val_small_20.parquet")

    #print(f"Number of rows in train_data: {train_data.count()}")
    #print(f"Number of rows in validation_data: {validation_data.count()}")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Split Data").getOrCreate()
    main()