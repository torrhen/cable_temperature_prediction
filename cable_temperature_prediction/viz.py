import matplotlib.pyplot as plt

def plot_loss(results: dict, model: str, folder_path):
  train_loss = results['train_loss']
  val_loss = results['val_loss']

  epochs = [(i + 1) for i in range(len(train_loss))]
  _, ax = plt.subplots(figsize=(12, 8))
  ax.plot(epochs, train_loss, label='Training loss')
  ax.plot(epochs, val_loss, label='Validation loss')
  ax.set_title(f"MSE loss of {model} for cable joint temperature prediction")
  ax.set_xlabel("Epochs")
  ax.set_ylabel("MSE Loss")
  plt.grid(True)
  plt.legend()
  plt.savefig(f"{str(folder_path)}/loss.png")
  plt.close()

def plot_predictions(y_pred, y_true, model: str, folder_path):
  plt.figure(figsize=(12, 8))
  plt.plot(y_pred.numpy(force=True), label='Predicted', alpha=0.8)
  if y_true is not None:
    plt.plot(y_true.numpy(force=True), label='True')
  plt.title(f"Cable joint temperature predictions of {model}")
  plt.xlabel('5 minute intervals')
  plt.grid(True)
  plt.legend()
  plt.savefig(f"{str(folder_path)}/predictions.png")
  plt.close()

def plot_error(y_pred, y_true, model: str, folder_path):
  errors = y_true - y_pred.squeeze()

  plt.figure(figsize=(12, 8))
  plt.plot(errors.numpy(force=True), label='prediction - real', c='r')
  plt.xlabel('5 minute intervals')
  plt.ylabel('Real vs predicted error')
  plt.title(f"Cable joint temperature prediction error of {model}")
  plt.grid(True)
  plt.legend()
  plt.savefig(f"{str(folder_path)}/error.png")
  plt.close()

