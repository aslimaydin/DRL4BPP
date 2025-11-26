import torch
import glob
import numpy as np
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
 
  def __init__(self, path, labels=None):
        'Initialization'
        self.path = path
        self.files = glob.glob(path + '/*.txt')

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.files)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        path =  self.files[index]
        print(path)
        #file = open(path, 'r')
        #data = int(file.readlines())
        # Load data and get label
        data=np.fromfile(path, dtype=int, sep=" \n").tolist()
        n_bins = data.pop(0)
        capacity = data.pop(0)
        bins = data
        return capacity,bins

if __name__ == "__main__":
    dataset = Dataset('./data/Falkenauer/Falkenauer_T')
    
    
    for i in range(dataset.__len__()):
        c,bins = dataset[i]
        print(i*1000+len(bins))