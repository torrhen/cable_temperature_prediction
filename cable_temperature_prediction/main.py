import torch
import config
import pandas as pd
from config import params
from models import get_model
from data import CableDataset
from engine import train, predict
from transforms import Transforms
from torch.utils.data import DataLoader
from viz import plot_loss, plot_predictions, plot_error


if __name__ == "__main__":

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Running on {device}")

  df = pd.read_csv(config.PROJECT_DATA_FILE_PATH, index_col=0, parse_dates=[0], dayfirst=True)
  df = df.resample("5min").mean()
  # replace nan values with forward fill of last observation
  df = df.interpolate()
  df = df.round(4)

  # 80:10:10 split
  train_df = df.iloc[:int(df.shape[0] * 0.8)]
  validation_df = df.iloc[int(df.shape[0] * 0.8):int(df.shape[0] * 0.9)]
  test_df = df.iloc[int(df.shape[0] * 0.9):]

  if not config.PROJECT_TRAIN_DATA_FILE_PATH.exists():
    train_df.to_csv(config.PROJECT_TRAIN_DATA_FILE_PATH)
  if not config.PROJECT_VALIDATION_DATA_FILE_PATH.exists():
    validation_df.to_csv(config.PROJECT_VALIDATION_DATA_FILE_PATH)
  if not config.PROJECT_TEST_DATA_FILE_PATH.exists():
    test_df.to_csv(config.PROJECT_TEST_DATA_FILE_PATH)

  model_params, train_params = params[config.MODEL_TYPE].values()

  # select model architecture
  model = get_model(config.MODEL_TYPE, model_params, device)

  # create train dataset
  train_data = CableDataset(
    config.PROJECT_TRAIN_DATA_FILE_PATH,
    model_params['window_size'],
    transforms=Transforms(),
  )
  train_loader = DataLoader(dataset=train_data, batch_size=train_params['batch_size'], shuffle=True, drop_last=False)

  train_data_mean, train_data_std, train_diffs_mean, train_diffs_std, train_stats_mean, train_stats_std = train_data.fit_normalisation()

  # create validation dataset and normalise with train statistics
  validation_data = CableDataset(
    config.PROJECT_VALIDATION_DATA_FILE_PATH,
    model_params['window_size'],
    transforms=None,
    normalise=(train_data_mean, train_data_std, train_diffs_mean, train_diffs_std, train_stats_mean, train_stats_std)
  )
  validation_loader = DataLoader(dataset=validation_data, batch_size=train_params['batch_size'], shuffle=True, drop_last=False)

  # create test dataset and normalise with train statistics
  test_data = CableDataset(
    config.PROJECT_TEST_DATA_FILE_PATH,
    model_params['window_size'],
    transforms=None,
    normalise=(train_data_mean, train_data_std, train_diffs_mean, train_diffs_std, train_stats_mean, train_stats_std)
  )
  test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=False)

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
  plot_loss(results, config.MODEL_TYPE, config.MODEL_OUTPUT_DIR)

  # load best model using model state
  best_model = get_model(config.MODEL_TYPE, model_params, device)
  best_model.load_state_dict(results['model_state'][best_epoch])

  # plot predictions and error for test data using best model
  y_pred, y_true = predict(best_model, test_loader, device=device)

  # concatenate predictions to test data frame
  test_df['Thermocouple 5 Predictions'] = test_df['Thermocouple 5']
  test_df['Thermocouple 5 Predictions'].iloc[model_params['window_size']:] = y_pred.numpy(force=True) # take the ground truth before predictions

  plot_predictions(test_df, config.MODEL_TYPE, config.MODEL_OUTPUT_DIR)
  plot_error(test_df, config.MODEL_TYPE, config.MODEL_OUTPUT_DIR)