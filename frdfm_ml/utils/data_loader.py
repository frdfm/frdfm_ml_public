import torch
from . import dataset_utils

def csv_data_loader(file_path, len, chunk_size=128 * 1024, loop_count=1, batch_size=32, **kwargs):
    csv_dataset = dataset_utils.CsvDataset(file_path, len, chunk_size=chunk_size, loop_count=loop_count, **kwargs)
    dataloader = torch.utils.data.DataLoader(csv_dataset, batch_size=batch_size)
    dataiter = iter(dataloader)
    return dataiter

