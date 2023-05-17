#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Script to get the 500 most popular songs and output this into a parquet file. 
Aware: Most popular is determined by number of different people who played a song, not total plays 
Usage:
    $ spark-submit pop_items.py <file_path>
'''

import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as func

def main(spark, file_path):
    '''
    Parameters
    ----------
    spark : SparkSession object
    file_path: original training data hdfs:/user/zz4140/1004-project-2023/interaction_count.parquet
    '''    
    
    # Loads the original parquet files
    track_train = spark.read.parquet(file_path)
    
    # Count total plays for tracks
    track_train.createOrReplaceTempView('track_train')
    #top_tracks = spark.sql('SELECT recording_msid, COUNT(*) as counts FROM track_train GROUP BY recording_msid ORDER BY count DESC LIMIT 500')
    top_tracks = (track_train
              .groupBy('recording_msid')
              .count()
              .orderBy(func.desc('count'))
              .limit(500))

    # Size = 500
    print("Total items in list:",top_tracks.count())
    print('')   
    
    # Check the first 15
    print('First 15:')
    top_tracks.limit(15).show() 
    
    # Save to a new parquet file
    top_track_ids = top_tracks.select(top_tracks.recording_msid)
    top_track_ids.write.parquet('items_popular.parquet')


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('popularTracks').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)