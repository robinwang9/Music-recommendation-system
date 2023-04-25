import pandas as pd

df = pd.read_parquet('/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet')

print(df.head(10))