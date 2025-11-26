import torch
import glob
import numpy as np
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
 
  def __init__(self, path=None, labels=None):
        'Initialization'
        self.path = path
        if path==None:
              self.files = glob.glob('.\data\**\*.txt', recursive=True)
        else:
            self.files = glob.glob(path + '/*.txt')
        #glob.glob('./data/**/*.txt', recursive=True)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.files)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        path =  self.files[index]
        data=np.fromfile(path, dtype=int, sep=" \n").tolist()
        n_bins = data.pop(0)
        capacity = data.pop(0)
        bins = data
        return capacity,bins

if __name__ == "__main__":
    dataset = Dataset()
    
    print(len(dataset))
    for i in range(len(dataset)):
        c,bins = dataset[i]