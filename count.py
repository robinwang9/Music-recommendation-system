from pyspark.sql import SparkSession

def read_parquet_file(parquet_file_path):

    spark = SparkSession.builder.appName("ReadParquetFile").getOrCreate()

    df = spark.read.parquet(parquet_file_path)

    spark.stop()

    return df