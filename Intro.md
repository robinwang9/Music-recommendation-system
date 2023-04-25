# Recommender Systems - Team 6 Introduction

A recommender system built in PySpark and designed for HPC environment. Submitted as term project for DS-GA 1004 Big Data at NYU.

This Introduction is written in Markdown and to give a brief overview of the project.

## Description of files:
- `Intro.md`: This file
- `README.md`: The original instruction file for the project
- `schema.py`: Find the schema of the parquet files
    > Usage: $ spark-submit --deploy-mode client schema.py <file_path>, 
    - Schema of 'tracks' file:
        - recording_msid: string (nullable = true)
        - artist_name: string (nullable = true)
        - track_name: string (nullable = true)
        - recording_mbid: string (nullable = true)
    
    - Schema of 'users' file:
        - user_id: integer (nullable = true)
        - user_name: string (nullable = true)

    - Schema of 'interactions' file:
        - user_id: integer (nullable = true)
        - recording_msid: string (nullable = true)
        - timestamp: timestamp (nullable = true)

