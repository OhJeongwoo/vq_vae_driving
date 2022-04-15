import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, t):
        self.t = t

    def __len__(self):
        return self.t
    
    def __getitem__(self, idx):
        return torch.LongTensor([idx]), 1.0

if __name__=="__main__":
    dataset = SimpleDataset(t=5)
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, drop_last=False)

    for batch in dataloader:
        print(batch[1])