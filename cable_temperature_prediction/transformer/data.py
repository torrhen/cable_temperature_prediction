import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CableDataset(Dataset):
  '''
  Medium voltage cable time series dataset. Derived from pytorch base class.
  '''
  def __init__(self, df, input_length, output_length, transforms=None, normalise=None):
    super().__init__()
    self.df = pd.read_csv(df)
    self.input_features = [
      'Thermocouple 1',
      'Thermocouple 2',
      'Thermocouple 3',
      'Thermocouple 4',
      'Thermocouple 6',
      'Thermocouple 7',
      'Phase (Blue)',
      'Phase (Yellow)',
      'Phase (Red)'
    ]
    self.target_feature = 'Thermocouple 5'
    self.input_length = input_length
    self.output_length = output_length
    self.transforms = transforms
    self.normalise = normalise
    self.X_data, self.X_diff, self.X_stats, self.y = self.make_dataset()

    # calulate the normalisation statistics from train data
    if not self.normalise:
      self.X_data_mean, self.X_data_std, self.X_diff_mean, self.X_diff_std, self.X_stats_mean, self.X_stats_std = self.fit_normalisation()
    else:
      self.X_data_mean, self.X_data_std, self.X_diff_mean, self.X_diff_std, self.X_stats_mean, self.X_stats_std = self.normalise
    
  def make_dataset(self):
    inputs, differences, statistics, targets = [], [], [], []

    for i in range(self.input_length, self.df.shape[0] - self.output_length):
      x = self.df.iloc[(i - self.input_length) : i][self.input_features].to_numpy()
      y = self.df.iloc[i : (i + self.output_length)][self.target_feature].to_numpy()

      d = self.calculate_differences(x)
      s = self.calculate_statistics(x)

      inputs.append(x)
      differences.append(d)
      statistics.append(s)
      targets.append(y)
    
    return np.stack(inputs), np.stack(differences), np.stack(statistics), np.stack(targets)
  
  
  def calculate_statistics(self, x):
    '''
    Calculate the summary statistics for each sample window.
    '''
    s = []
    s.append(np.amin(x, axis=0))
    s.append(np.amax(x, axis=0))
    s.append(np.median(x, axis=0))
    s.append(np.mean(x, axis=0))
    s.append(np.var(x, axis=0))
    return np.concatenate(s)
  
  def calculate_differences(self, x):
    '''
    Calculate the first differences between consecutive time interval samples within each window.
    '''
    d = np.diff(x, axis=0)
    return np.vstack([np.zeros_like(d[0]), d])

  def fit_normalisation(self):
    # return the mean and standard deviation for window samples and summary staticstics over whople dataset separately
    return np.mean(self.X_data, axis=(0, 1)), np.std(self.X_data, axis=(0, 1)), np.mean(self.X_diff, axis=(0, 1)), np.std(self.X_diff, axis=(0, 1)), np.mean(self.X_stats, axis=0), np.std(self.X_stats, axis=0)
  
  def __len__(self):
    return self.X_data.shape[0]

  def __getitem__(self, idx):
    x_tensor = torch.as_tensor(self.X_data[idx], dtype=torch.float32)
    y_tensor = torch.as_tensor(self.y[idx], dtype=torch.float32).unsqueeze(dim=-1)

    # augment new samples
    if self.transforms:
      x_tensor, y_tensor = self.transforms((x_tensor, y_tensor))

    # calculate new summary stats on window
    s_tensor = torch.as_tensor(self.calculate_statistics(x_tensor.numpy(force=True)), dtype=torch.float32)
    # calculate differences of samples within window
    d_tensor = torch.as_tensor(self.calculate_differences(x_tensor.numpy(force=True)), dtype=torch.float32)

    # normalise data and summary statistics based on training data
    x_tensor = (x_tensor - self.X_data_mean) / (self.X_data_std + 1e-8)
    d_tensor = (d_tensor - self.X_diff_mean) / (self.X_diff_std + 1e-8)
    s_tensor = (s_tensor - self.X_stats_mean) / (self.X_stats_std + 1e-8)

    # clip values to +/- 3 stdd
    x_tensor = torch.clip(x_tensor, min=-3.0, max=3.0)
    d_tensor = torch.clip(d_tensor, min=-3.0, max=3.0)
    s_tensor = torch.clip(s_tensor, min=-3.0, max=3.0)

    x_tensor = torch.concatenate([x_tensor, d_tensor], axis=1)

    return x_tensor, s_tensor, y_tensor
  
