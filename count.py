import sys
from pyspark.sql import SparkSession

if __name__ == "__main__":
    # Create Spark session
    spark = SparkSession.builder.appName("CountRows").getOrCreate()

    # Read parquet file
    parquet_file_path = sys.argv[1]
    df = spark.read.parquet(parquet_file_path)

    # Count rows
    num_rows = df.count()

    # Print result
    print(f"Number of rows in {parquet_file_path}: {num_rows}")

    # Stop Spark session
    spark.stop()