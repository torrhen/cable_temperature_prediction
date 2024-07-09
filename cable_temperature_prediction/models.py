import torch
from torch import nn

# custom RNN model
class RecurrentNeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(RecurrentNeuralNetwork, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.output_size = output_size

    self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, nonlinearity='relu', batch_first=True)
    self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

  def forward(self, x):
    batch_size = x.shape[0]
    # reset hidden state for each new batch
    hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

    output, hidden = self.rnn(x, hidden_state.detach())
    # pass hidden state of final sequence element for each training sample in batch to linear layer
    output = self.fc(output[:, -1, :])

    return output


class LongShortTermMemoryNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(LongShortTermMemoryNetwork, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.output_size = output_size

    self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
    self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

  def forward(self, x):
    batch_size = x.shape[0]
    # reset hidden state and cell state for each new batch
    self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
    self.cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

    output, hidden = self.lstm(x, (self.hidden_state.detach(), self.cell_state.detach()))
    # pass hidden state of final sequence element for each training sample in batch to linear layer
    output = self.fc(output[:, -1, :])

    return output


class GatedRecurrentUnitNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(GatedRecurrentUnitNetwork, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.output_size = output_size

    self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
    self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)


  def forward(self, x):
    batch_size = x.shape[0]
    # reset the hidden state for each batch
    self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

    output, hidden = self.gru(x, self.hidden_state.detach())
    # pass hidden state of final sequence element for each training sample in batch to linear layer
    output = self.fc(output[:, -1, :])

    return output