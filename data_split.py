def data_subsampling(spark, partial):
    '''
    Subsample the data to get a smaller dataset
    
    This function returns a dataframe corresponding to training, validation

    Parameters
    ----------
    spark : spark session object
    partial: float
        The subsampling rate
    '''
    # This loads the parquet file
    interactions = spark.read.parquet('hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet')

    # Subsample the data
    interactions = interactions.sample(False, partial, seed=100)
    interactions.createOrReplaceTempView('interactions')

    # Split the data into training, validation
    train, validation = interactions.randomSplit([0.8, 0.2], seed=100)
    train.createOrReplaceTempView('train')
    validation.createOrReplaceTempView('validation')

    #return train, validation