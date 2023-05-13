import pyarrow.parquet as pq
import sys

def count_rows(parquet_file_path):
    parquet_file = pq.ParquetFile(parquet_file_path)

    num_rows = parquet_file.metadata.num_rows

    print("Number of rows in the Parquet file:", num_rows)

if __name__ == '__main__':
    file_path = sys.argv[1]

    count_rows(file_path)