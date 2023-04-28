#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Example python script to run benchmark on a query with a file path.
Usage:
    $ spark-submit --deploy-mode client repartition_check.py <file_path>
'''


# Import command line arguments and helper functions
import sys
#import bench

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession


def main(spark, file_path):
    '''Construct a basic query on the people dataset

    This function returns a dataframe corresponding to the
    first five people, ordered alphabetically by last_name, first_name.

    Parameters
    ----------
    spark : spark session object

    file_path : string
        The path (in HDFS) to the CSV file, e.g.,
        `hdfs:/user/pw44_nyu_edu/peopleSmall.csv`

    Returns
    top5:
        Uncomputed dataframe of first 5 people in dataset
    '''
    '''
    # This loads the parquet file with proper header decoding and schema
    data = spark.read.parquet(file_path, header=True, inferSchema=True)
                            #schema='first_name STRING, last_name STRING, age INT, income FLOAT, zipcode INT, orders INT, loyalty BOOLEAN, rewards BOOLEAN')

    data.createOrReplaceTempView('data')

    top5 = spark.sql('SELECT * FROM data LIMIT 5')

    print('Top 5 line:')
    print(top5)

    '''
    spark = SparkSession.builder.appName("ReadParquetFile").getOrCreate()
    parquet_file = spark.read.parquet(file_path)

    print(parquet_file.head())
    print("Number of rows: ", parquet_file.count())


'''
def main(spark, file_path):

    # Example Benchmarking Query
    times = bench.benchmark(spark, 5, basic_query, file_path)

    print(f'Times to run Basic Query 5 times on {file_path}')
    print(times)
    print(f'Maximum Time taken to run Basic Query 5 times on {file_path}:{max(times)}')

    # You can do list calculations for your analysis here!
'''

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part2').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)