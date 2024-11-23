from . import generators as gen
import torch
from torch.utils.data import Dataset

class CsvDataset(Dataset):
    def __init__(self, file_path, chunk_size=128 * 1024, loop_count=999_999_999_999_999_999_999_999_999_999, **kwargs):
        self.csv_generator = gen.csv_generator(file_path, chunk_size=chunk_size, loop_count=loop_count)
        self.kwargs = kwargs

    def __len__(self):
        return 999_999_999_999_999

    def __getitem__(self, idx):
        passed = False
        while not passed:
            try:
                next_row = next(self.csv_generator)
                if 'etl_func' in self.kwargs:
                    next_row = self.kwargs['etl_func'](next_row, **self.kwargs)
                passed = True
            except:
                pass
        return next_row
