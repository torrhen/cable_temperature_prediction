import os
import torch
import pandas as pd
from pathlib import Path
from model import Transformer
from data import CableDataset
from engine import train, predict
from transforms import Transforms
from torch.utils.data import DataLoader
from viz import plot_loss, plot_predictions, plot_error


model_params = {
  'input_length' : 12,
  'output_length' : 12,
  'num_encoder_layers' : 1,
  'num_decoder_layers' : 1,
  'embedding_dim' : 64,
  'num_heads' : 2,
  'hidden_dims' : 32
}

train_params = {
  'epochs' : 50,
  'batch_size' : 32,
  'lr' : 1e-4,
  'weight_decay' : 0.01
}


if __name__ == "__main__":

  PROJECT_ROOT_DIR = Path(os.getcwd())

  PROJECT_DATA_DIR = PROJECT_ROOT_DIR / Path("data")
  PROJECT_DATA_FILE_PATH = PROJECT_DATA_DIR / Path("cable.csv")
  PROJECT_TRAIN_DATA_FILE_PATH = PROJECT_DATA_DIR / Path("train.csv")
  PROJECT_VALIDATION_DATA_FILE_PATH = PROJECT_DATA_DIR / Path("validation.csv")
  PROJECT_TEST_DATA_FILE_PATH = PROJECT_DATA_DIR / Path("test.csv")

  PROJECT_SCRIPT_DIR = PROJECT_ROOT_DIR / Path("cable_temperature_prediction")

  PROJECT_OUTPUT_DIR = PROJECT_SCRIPT_DIR / Path("transformer/output")
  PROJECT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Running on {device}")

  df = pd.read_csv(PROJECT_DATA_FILE_PATH, index_col=0, parse_dates=[0], dayfirst=True)
  df = df.resample("5min").mean()
  # replace nan values with forward fill of last observation
  df = df.interpolate()
  df = df.round(4)

  # 80:10:10 split
  train_df = df.iloc[:int(df.shape[0] * 0.8)]
  validation_df = df.iloc[int(df.shape[0] * 0.8):int(df.shape[0] * 0.9)]
  test_df = df.iloc[int(df.shape[0] * 0.9):]

  if not PROJECT_TRAIN_DATA_FILE_PATH.exists():
    train_df.to_csv(PROJECT_TRAIN_DATA_FILE_PATH)
  if not PROJECT_VALIDATION_DATA_FILE_PATH.exists():
    validation_df.to_csv(PROJECT_VALIDATION_DATA_FILE_PATH)
  if not PROJECT_TEST_DATA_FILE_PATH.exists():
    test_df.to_csv(PROJECT_TEST_DATA_FILE_PATH)


  # create train dataset
  train_data = CableDataset(
    PROJECT_TRAIN_DATA_FILE_PATH,
    model_params['input_length'],
    model_params['output_length'],
    transforms=Transforms(),
  )
  train_loader = DataLoader(dataset=train_data, batch_size=train_params['batch_size'], shuffle=True, drop_last=False)

  train_data_mean, train_data_std, train_diffs_mean, train_diffs_std, train_stats_mean, train_stats_std = train_data.fit_normalisation()

  # create validation dataset and normalise with train statistics
  validation_data = CableDataset(
    PROJECT_VALIDATION_DATA_FILE_PATH,
    model_params['input_length'],
    model_params['output_length'],
    transforms=None,
    normalise=(train_data_mean, train_data_std, train_diffs_mean, train_diffs_std, train_stats_mean, train_stats_std)
  )
  validation_loader = DataLoader(dataset=validation_data, batch_size=train_params['batch_size'], shuffle=True, drop_last=False)

  # create test dataset and normalise with train statistics
  test_data = CableDataset(
    PROJECT_TEST_DATA_FILE_PATH,
    model_params['input_length'],
    model_params['output_length'],
    transforms=None,
    normalise=(train_data_mean, train_data_std, train_diffs_mean, train_diffs_std, train_stats_mean, train_stats_std)
  )
  test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=False)

  # model architecture
  model = Transformer(
    input_length=model_params['input_length'],
    num_encoder_layers=model_params['num_encoder_layers'],
    num_decoder_layers=model_params['num_decoder_layers'],
    embedding_dim=model_params['embedding_dim'],
    num_heads=model_params['num_heads'],
    hidden_dims=model_params['hidden_dims']
  ).to(device)

  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])

  results = train(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=validation_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    epochs=train_params['epochs']
  )

  best_epoch = results['val_loss'].index(min(results['val_loss'])) # best epoch based on validation loss
  best_train_loss = results['train_loss'][best_epoch]
  best_validation_loss = results['val_loss'][best_epoch]

  print(f"\nBest Epoch: {best_epoch + 1}")
  print("----------")
  print(f"| Best Train Loss: {results['train_loss'][best_epoch]:.4f} |\n")
  print(f"| Best Validation Loss: {results['val_loss'][best_epoch]:.4f} |\n")

  # plot train and validation loss curves
  plot_loss(results, "transformer", PROJECT_OUTPUT_DIR)

  # load best model using model state
  best_model = Transformer(
    input_length=model_params['input_length'],
    num_encoder_layers=model_params['num_encoder_layers'],
    num_decoder_layers=model_params['num_decoder_layers'],
    embedding_dim=model_params['embedding_dim'],
    num_heads=model_params['num_heads'],
    hidden_dims=model_params['hidden_dims']
  ).to(device)

  best_model.load_state_dict(results['model_state'][best_epoch])

  # plot predictions and error for test data using best model
  y_pred, y_true = predict(best_model, test_loader, device=device)

  # concatenate predictions to test data frame
  test_df['Thermocouple 5 Predictions'] = test_df['Thermocouple 5']
  # keep some number of the first and last true values as a result of spliting data into sequences
  test_df['Thermocouple 5 Predictions'].iloc[model_params['input_length']:-model_params['output_length']] = y_pred.numpy(force=True)

  plot_predictions(test_df, "transformer", PROJECT_OUTPUT_DIR)
  plot_error(test_df, "transformer", PROJECT_OUTPUT_DIR)