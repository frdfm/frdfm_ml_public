import pandas as pd

def csv_generator(file_path, chunk_size=128 * 1024, loop_count=1):
    for _ in range(loop_count):
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                yield row