import matplotlib.pyplot as plt


def plot_loss(results: dict, model: str, folder_path):
  '''
  Plot the training and validation loss curves.
  '''
  train_loss = results['train_loss']
  val_loss = results['val_loss']

  epochs = [(i + 1) for i in range(len(train_loss))]
  _, ax = plt.subplots(figsize=(12, 8))
  ax.plot(epochs, train_loss, label='Training loss')
  ax.plot(epochs, val_loss, label='Validation loss')
  ax.set_title(f"MSE loss of {model} for cable joint temperature (Thermocouple 5) predictions")
  ax.set_xlabel("Epochs")
  ax.set_ylabel("MSE Loss")
  plt.grid(True)
  plt.legend()
  plt.savefig(f"{str(folder_path)}/loss.png")
  plt.close()


def plot_predictions(test_df, model: str, folder_path):
  '''
  Plot the ground truth and predictions on the test data
  '''
  ax = test_df[['Thermocouple 5', 'Thermocouple 5 Predictions']].plot(figsize=(20, 8))
  ax.set_title(f"{model} predictions of cable joint temperature (Thermocouple 5)")
  ax.set_ylabel('Temperature (C)')
  plt.grid(True)
  plt.legend()
  plt.savefig(f"{str(folder_path)}/predictions.png")
  plt.close()


def plot_error(test_df, model: str, folder_path):
  '''
  Plot the error of predictions on the test data
  '''
  test_df['Thermocouple 5 Error'] = test_df['Thermocouple 5 Predictions'] - test_df['Thermocouple 5']
  ax = test_df['Thermocouple 5 Error'].plot(figsize=(20, 8), color='r')
  ax.set_title(f"{model} prediction error of cable joint temperature (Thermocouple 5)")
  ax.set_ylabel('Error')
  plt.grid(True)
  plt.legend()
  plt.savefig(f"{str(folder_path)}/error.png")
  plt.close()