import os
import torch
from pathlib import Path
from config import params
from engine import train, predict
from torch.utils.data import DataLoader
from viz import plot_loss, plot_predictions, plot_error
from data import read_data, moving_window_normalization, CableSeriesData, Augmentations
from models import RecurrentNeuralNetwork, LongShortTermMemoryNetwork, GatedRecurrentUnitNetwork


if __name__ == "__main__":

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(device)

  SAMPLES_PER_HOUR = 12
  HOURS_IN_DAY = 24

  # MODEL_TYPE = 'RNN'
  # MODEL_TYPE = 'LSTM'
  MODEL_TYPE = 'GRU'

  DATA_FILE_PATH = Path(os.getcwd()) / Path("data/cable.csv")
  df = read_data(DATA_FILE_PATH)
  df = moving_window_normalization(df, window_size=SAMPLES_PER_HOUR * HOURS_IN_DAY)

  train_df = df.iloc[:-(SAMPLES_PER_HOUR * HOURS_IN_DAY * 3)].round(4)
  test_df = df.iloc[-(SAMPLES_PER_HOUR * HOURS_IN_DAY * 3):].round(4)

  OUTPUT_FOLDER_PATH = Path(os.getcwd()) / Path(f"output/{MODEL_TYPE.lower()}")
  OUTPUT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

  data_params, model_params, train_params = params[MODEL_TYPE].values()

  if MODEL_TYPE == "RNN":
    model = RecurrentNeuralNetwork(input_size=1, hidden_size=model_params['hidden_size'], num_layers=model_params['num_layers'], output_size=data_params['output_length']).to(device)

  elif MODEL_TYPE == "LSTM":
    model = LongShortTermMemoryNetwork(input_size=1, hidden_size=model_params['hidden_size'], num_layers=model_params['num_layers'], output_size=1).to(device)

  elif MODEL_TYPE == "GRU":
    model = GatedRecurrentUnitNetwork(input_size=1, hidden_size=model_params['hidden_size'], num_layers=model_params['num_layers'], output_size=1).to(device)
  else:
    raise NotImplementedError

  # isolate thermocouple of cable joint
  train_df = train_df['Thermocouple 5']
  test_df = test_df['Thermocouple 5']

  train_data = CableSeriesData(train_df, data_params['input_length'], data_params['output_length'], transforms=Augmentations())
  test_data = CableSeriesData(test_df, data_params['input_length'], data_params['output_length'], transforms=None)
  
  train_loader = DataLoader(dataset=train_data, batch_size=train_params['batch_size'], shuffle=False, drop_last=False)
  test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=False)

  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=train_params['lr'])

  results = train(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=test_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    epochs=train_params['epochs']
  )

  y_pred, y_true = predict(model, test_loader, device=device)

  plot_loss(results, MODEL_TYPE, OUTPUT_FOLDER_PATH)
  plot_predictions(y_pred, y_true, MODEL_TYPE, OUTPUT_FOLDER_PATH)
  plot_error(y_pred, y_true, MODEL_TYPE, OUTPUT_FOLDER_PATH)    




  

