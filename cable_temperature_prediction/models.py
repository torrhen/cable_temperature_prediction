import torch
import config
from torch import nn


class RecurrentNeuralNetwork(nn.Module):
  '''
  Recurrent Neural Network.
  Window sequence data is given to RNN and concatenated with summary statistics before being given to fully connected layer.
  '''
  def __init__(self, in_features, hidden_size, num_layers, out_features):
    super(RecurrentNeuralNetwork, self).__init__()
    self.in_features = in_features
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.out_features = out_features
    self.rnn = nn.RNN(input_size=self.in_features, hidden_size=self.hidden_size, num_layers=self.num_layers, nonlinearity='relu', batch_first=True, dropout=0.1)
    self.fc = nn.LazyLinear(out_features=1)

  def forward(self, x, s):
    batch_size = x.shape[0]
    # reset hidden state for each new batch
    hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

    output, hidden = self.rnn(x.float(), hidden_state.detach())
    # pass hidden state of final sequence element for each training sample in batch to linear layer
    output = torch.concat([output[:, -1, :], s], axis=1)
    output = self.fc(output.float())

    return output


class LongShortTermMemoryNetwork(nn.Module):
  '''
  Long Short Term Memory Network for better handling of vanishing gradients and longer temporal sequence lengths.
  Window sequence data is given to LSTM and concatenated with summary statistics before being given to fully connected layer.
  '''
  def __init__(self, in_features, hidden_size, num_layers, out_features):
    super(LongShortTermMemoryNetwork, self).__init__()
    self.in_features = in_features
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.out_features = out_features
    self.lstm = nn.LSTM(input_size=self.in_features, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=0.1)
    self.fc = nn.LazyLinear(out_features=self.out_features)

  def forward(self, x, s):
    batch_size = x.shape[0]
    # reset hidden state and cell state for each new batch
    self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
    self.cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

    output, hidden = self.lstm(x.float(), (self.hidden_state.detach(), self.cell_state.detach()))
    # pass hidden state of final sequence element for each training sample in batch to linear layer
    output = torch.concat([output[:, -1, :], s], axis=1)
    output = self.fc(output.float())

    return output


class GatedRecurrentUnitNetwork(nn.Module):
  '''
  Gated Recurrent Unit Network. More efficient architecture than LSTM to reduce potential for overfitting.
  Window sequence data is given to GRU and concatenated with summary statistics before being given to fully connected layer.
  '''
  def __init__(self, in_features, hidden_size, num_layers, out_features):
    super(GatedRecurrentUnitNetwork, self).__init__()
    self.in_features = in_features
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.out_features = out_features
    self.gru = nn.GRU(self.in_features, self.hidden_size, self.num_layers, batch_first=True, dropout=0.1)
    self.fc = nn.LazyLinear(out_features=self.out_features)

  def forward(self, x, s):
    batch_size = x.shape[0]
    # reset the hidden state for each batch
    self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

    output, hidden = self.gru(x.float(), self.hidden_state.detach())
    # pass hidden state of final sequence element for each training sample in batch to linear layer
    output = torch.concat([output[:, -1, :], s], axis=1)
    output = self.fc(output.float())

    return output
  

def get_model(model:str, params, device):
  '''
  Select the model architecture and hyper parameters for training.
  '''
  # select and build model
  if config.MODEL_TYPE == "RNN":
    model = RecurrentNeuralNetwork(
      in_features=params['in_features'],
      hidden_size=params['hidden_size'],
      num_layers=params['num_layers'],
      out_features=params['out_features']
    ).to(device)

  elif config.MODEL_TYPE == "LSTM":
    model = LongShortTermMemoryNetwork(
      in_features=params['in_features'],
      hidden_size=params['hidden_size'],
      num_layers=params['num_layers'],
      out_features=params['out_features']
    ).to(device)

  elif config.MODEL_TYPE == "GRU":
    model = GatedRecurrentUnitNetwork(
      in_features=params['in_features'],
      hidden_size=params['hidden_size'],
      num_layers=params['num_layers'],
      out_features=params['out_features']
    ).to(device)

  else:
    raise NotImplementedError
  
  return model