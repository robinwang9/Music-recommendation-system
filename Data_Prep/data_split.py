'''
import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from tqdm import tqdm
'''
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import count, row_number, rand
from pyspark.sql.functions import round
from tqdm import tqdm

#Usage: spark-submit --deploy-mode client data_split.py

def tqdm_count_rows(iterator):
    count = 0
    for row in tqdm(iterator, desc='Counting rows'):
        count += 1
    return [count]


def main(spark):
    df = spark.read.parquet("hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet")

    # Subsample the data
    df = df.sample(False, 0.5, seed=42)

    counts = df.groupBy("user_id").count()

    valid_user_ids = counts.filter(counts["count"] >= 100).select("user_id")

    cleaned_df = df.join(valid_user_ids, "user_id")

    user_counts = cleaned_df.groupBy("user_id").count()

    user_counts = user_counts.withColumn("rounded_count", round(user_counts["count"]))

    user_counts_list = user_counts.collect()

    sampling_fractions = {row["user_id"]: row["rounded_count"]*0.8/row["count"] for row in user_counts_list}

    train_df = cleaned_df.sampleBy("user_id", fractions=sampling_fractions, seed=42)
    val_df = cleaned_df.subtract(train_df)

    # Apply tqdm progress bar to count rows in train and validation DataFrames
    train_row_count = train_df.rdd.mapPartitions(tqdm_count_rows).sum()
    validation_row_count = val_df.rdd.mapPartitions(tqdm_count_rows).sum()

    print(f'Train row count: {train_row_count}')
    print(f'Validation row count: {validation_row_count}')

    train_df.write.parquet("interactions_train_test.parquet")
    val_df.write.parquet("interactions_val_test.parquet")
    
    return train_df, val_df

    spark.stop()


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Split Data").getOrCreate()
    train_df, val_df = main(spark)
