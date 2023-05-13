# Import command line arguments and helper functions
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count


def main(spark, input_file):
    '''Construct a basic query on the people dataset

    This function returns a dataframe contains user_id, recording_msid, and count(times per user_id listen to different track).

    Usage: `$ spark-submit --deploy-mode client count.py`

    Parameters
    ----------
    spark : spark session object

    file_path : string
        The path (in HDFS) to the CSV file, e.g.,
        `hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet`

    Returns : 
        Dataframe contains user_id, recording_msid, and count(times per user_id listen to different track).
    '''
    # This loads the Parquet file with proper header decoding and schema
    interactions_df = spark.read.parquet(input_file)
    interactions_df.createOrReplaceTempView('interactions_df')

    # Count interactions
    num_rows = interactions_df.metadata.num_rows

    print("Number of rows in the Parquet file:", num_rows)
       


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('InteractionsCount').getOrCreate()

    # Get file_path for dataset to analyze
    # file_path = sys.argv[1]

    main(spark, input_file)