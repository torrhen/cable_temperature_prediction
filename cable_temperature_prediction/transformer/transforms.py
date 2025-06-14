import torch
import torchvision


class GaussianNoise(torch.nn.Module):
  '''
  Perturb multivariate sequence data with some (very small) amount of gaussian noise.
  '''
  def __init__(self):
    super().__init__()

  def forward(self, sample):
    x, y = sample
    return x + torch.randn(x.shape) * 0.01, y


class PermuteSequence(torch.nn.Module):
  '''
  Shuffle the order of some small, random subset of consecutive time intervals.
  '''
  def __init__(self, size=3):
    super().__init__()
    self.size = size

  def forward(self, sample):
    x, y = sample

    # choose a random time interval from within the sequence
    r_idx = torch.randint(low=0, high=(x.shape[0] - self.size), size=(1,))
    # store the sample of data based on the time interval and number of rows to permute
    _x = x[r_idx:r_idx + self.size, :]
    # shuffle time intervals of subset rows
    _x = _x[torch.randperm(self.size), :]
    # overwrite the original rows with the shuffled rows
    x[r_idx:r_idx + self.size, :] = _x

    return x, y


# class FreezeSequence(torch.nn.Module):
#   '''
#   Duplicate a random time interval for a random number of future, consecutive time intervals. Not used.
#   '''
#   def __init__(self, size=3):
#     super().__init__()
#     self.size = size

#   def forward(self, sample):
#     x, y = sample
#     # choose a random time interval from within the sequence
#     r_idx = torch.randint(low=0, high=(x.shape[0] - self.size), size=(1,))
#     # overwrite the original rows with the duplicated rows
#     x[r_idx:r_idx + self.size, :] = x[r_idx, :].repeat(self.size, 1)

#     return x, y
  

class ScalingSequence(torch.nn.Module):
  '''
  Scale features and target by some normal random scalar.
  '''
  def __init__(self):
    super().__init__()

  def forward(self, sample):
    x, y = sample
    scale = torch.normal(mean=1, std=0.05, size=(1,))
    # scale = torch.FloatTensor(1).uniform_(0.8, 1.2)

    return x * scale, y * scale
  

class Transforms(torch.nn.Module):
  '''
  Augmentation pipeline.
  '''
  def __init__(self):
    super().__init__()

  def forward(self, sample):
    return torch.nn.Sequential(
      torchvision.transforms.RandomApply([GaussianNoise()], p=0.5),
      torchvision.transforms.RandomApply([ScalingSequence()], p=0.5),
      torchvision.transforms.RandomApply([PermuteSequence()], p=0.5),
    )(sample)


