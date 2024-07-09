import matplotlib.pyplot as plt

def plot_loss(results: dict, folder_path):
  train_loss = results['train_loss']
  val_loss = results['val_loss']

  epochs = [(i + 1) for i in range(len(train_loss))]
  _, ax = plt.subplots(figsize=(12, 8))
  ax.plot(epochs, train_loss, label='Training loss')
  ax.plot(epochs, val_loss, label='Validation loss')
  ax.set_title("RNN loss on cable time series data")
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Loss")
  plt.grid(True)
  plt.legend()
  plt.savefig(f"{str(folder_path)}/loss.png")
  plt.close()

def plot_predictions(y_pred, y_true, folder_path):
  plt.figure(figsize=(12, 8))
  plt.plot(y_pred.numpy(force=True), label='Predicted', alpha=0.8)
  if y_true is not None:
    plt.plot(y_true.numpy(force=True), label='True')
  plt.xlabel('5 minute intervals')
  plt.grid(True)
  plt.legend()
  plt.savefig(f"{str(folder_path)}/predictions.png")
  plt.close()

def plot_error(y_pred, y_true, folder_path):
  # calculate error
  errors = y_true - y_pred.squeeze()

  plt.figure(figsize=(12, 8))
  plt.plot(errors.numpy(force=True), label='error', c='r')
  plt.xlabel('5 minute intervals')
  plt.ylabel('Real vs predicted error')
  plt.grid(True)
  plt.legend()
  plt.savefig(f"{str(folder_path)}/error.png")
  plt.close()

