from . import generators as gen
import torch
from torch.utils.data import Dataset

class CsvDataset(Dataset):
    def __init__(self, file_path, len, chunk_size=128 * 1024, loop_count=1, **kwargs):
        self.csv_generator = gen.csv_generator(file_path, chunk_size=chunk_size, loop_count=loop_count)
        self.kwargs = kwargs
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        passed = False
        while not passed:
            try:
                next_row = next(self.csv_generator)
                if 'etl_func' in self.kwargs:
                    next_row = self.kwargs['etl_func'](next_row, **self.kwargs)
                passed = True
            except StopIteration:
                raise StopIteration
            except Exception:
                continue
        return next_row


class CsvDataset_from_two_files(Dataset):
    def __init__(self, file_path1, file_path2, len, chunk_size=128 * 1024, loop_count=1, **kwargs):
        self.csv_generator = gen.csv_generator_from_two_files(file_path1, file_path2, chunk_size=chunk_size, loop_count=loop_count)
        self.kwargs = kwargs
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        passed = False
        while not passed:
            try:
                next_row = next(self.csv_generator)
                if 'etl_func' in self.kwargs:
                    next_row = self.kwargs['etl_func'](next_row, **self.kwargs)
                passed = True
            except StopIteration:
                raise StopIteration
            except Exception:
                continue
        return next_row
