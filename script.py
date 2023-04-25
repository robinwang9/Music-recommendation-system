import pandas as pd

df = pd.read_parquet('hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet')

print(df.head(10))