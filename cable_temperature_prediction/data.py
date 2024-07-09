import torch
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def read_data(filepath: str):
  df = pd.read_csv(filepath, index_col=0, parse_dates=[0], dayfirst=True)
  df = df.resample("5min").mean()
  # replace nan values with forward fill of last observation
  df = df.interpolate().round(2)
  return df

def moving_window_normalization(df: pd.DataFrame, window_size: int):
  _df = df.iloc[window_size:, :].copy()
  for i in range(window_size, df.shape[0]):
    _df.iloc[i - window_size, :] = (((df.iloc[i - window_size : i] - df.iloc[i - window_size : i].mean(axis=0)) / (df.iloc[i - window_size : i].std(axis=0) + 1e-5)).iloc[-1, :])
  return _df

class GaussianNoise(torch.nn.Module):
  '''
  Perturb sequence data with some amount of gaussian noise.
  '''
  def __init__(self, var=0.1):
    super().__init__()
    self.var = var

  def forward(self, x):
    return x + (self.var**0.5) * torch.randn(x.shape)

class SequenceMask(torch.nn.Module):
  '''
  Randomly mask out data within each sequence.
  '''
  def __init__(self, p=0.2):
    super().__init__()
    self.p = p
    self.dropout = torch.nn.Dropout1d(p=self.p)

  def forward(self, x):
    with torch.inference_mode():
      return self.dropout(x)

class Augmentations(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.transforms = torch.nn.Sequential(
      torchvision.transforms.RandomApply([GaussianNoise()], p=0.5),
      torchvision.transforms.RandomApply([SequenceMask()], p=0.5)
    )
  
  def forward(self, x):
    return self.transforms(x)

class CableSeriesData(Dataset):
  def __init__(self, df, input_length, output_length, transforms=None):
    super().__init__()
    self.df = df
    self.input_length = input_length
    self.output_length = output_length
    self.transforms = transforms
    self.X, self.y = self.make_dataset()

  def make_dataset(self):
    inputs, targets = [], []
    for i in range(self.df.shape[0] - (self.input_length + self.output_length)):
      inputs.append(self.df[i : i + self.input_length].values)
      targets.append(self.df[i + self.input_length : i + (self.input_length + self.output_length)].values)
    return np.stack(inputs), np.stack(targets)

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    x, y = torch.as_tensor(self.X[idx], dtype=torch.float32).unsqueeze(dim=-1), torch.as_tensor(self.y[idx], dtype=torch.float32)
    if self.transforms:
      x = self.transforms(x)
    return x, y
  
