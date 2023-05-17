#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Example python script to run benchmark on a query with a file path.
Usage:
    $ spark-submit --deploy-mode client schema.py <file_path>
'''


# Import command line arguments and helper functions
import sys
#import bench

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession


def main(spark, file_path):

    spark = SparkSession.builder.appName("ReadParquetFile").getOrCreate()
    parquet_file = spark.read.parquet(file_path)
    parquet_file.printSchema()



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part2').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)