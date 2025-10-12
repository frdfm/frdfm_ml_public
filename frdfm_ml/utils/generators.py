import pandas as pd

def csv_generator(file_path, chunk_size=128 * 1024, loop_count=1):
    for _ in range(loop_count):
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                yield row
import csv

def csv_generator_from_two_files(file_path1, file_path2, chunk_size=128 * 1024, loop_count=1):
    for _ in range(loop_count):
        reader1 = pd.read_csv(file_path1, chunksize=chunk_size, header=None, sep='\x1e')
        reader2 = pd.read_csv(file_path2, chunksize=chunk_size, header=None, sep='\x1e')

        for chunk1, chunk2 in zip(reader1, reader2):
            # iterate row-wise in parallel
            for (_, row1), (_, row2) in zip(chunk1.iterrows(), chunk2.iterrows()):
                yield [row1[0], row2[0]]