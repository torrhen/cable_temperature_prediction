import os
from pathlib import Path


PROJECT_ROOT_DIR = Path(os.getcwd())

PROJECT_DATA_DIR = PROJECT_ROOT_DIR / Path("data")

PROJECT_DATA_FILE_PATH = PROJECT_DATA_DIR / Path("cable.csv")

PROJECT_TRAIN_DATA_FILE_PATH = PROJECT_DATA_DIR / Path("train.csv")

PROJECT_VALIDATION_DATA_FILE_PATH = PROJECT_DATA_DIR / Path("validation.csv")

PROJECT_TEST_DATA_FILE_PATH = PROJECT_DATA_DIR / Path("test.csv")

PROJECT_OUTPUT_DIR = PROJECT_ROOT_DIR / Path("output")

# select the model to train and evaulate
MODEL_TYPE = 'RNN'
# MODEL_TYPE = 'LSTM'
# MODEL_TYPE = 'GRU'

MODEL_OUTPUT_DIR = PROJECT_OUTPUT_DIR / Path(f"{MODEL_TYPE.lower()}")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# architecture and training hyperparameters
params = {
  'RNN' : {
    'model' : {
      'in_features' : 18,
      'window_size' : 12, # 1 hour
      'hidden_size' : 16,
      'out_features' : 1,
      'num_layers' : 1          
    },
    'train' : {
      'epochs' : 50,
      'batch_size' : 32,
      'lr' : 1e-4,
      'weight_decay' : 0.01
    }
  },
  'LSTM' : {
    'model' : {
      'in_features' : 18,
      'window_size' : 12, # 1 hour
      'hidden_size' : 16,
      'cell_size' : 16,
      'out_features' : 1,
      'num_layers' : 1        
    },
    'train' : {
      'epochs' : 50,
      'batch_size' : 32,
      'lr' : 1e-4,
      'weight_decay' : 0.01
    }
  },
  'GRU' : {
    'model' : {
      'in_features' : 18,
      'window_size' : 12, # 1 hour
      'hidden_size' : 16,
      'out_features' : 1,
      'num_layers' : 1        
    },
    'train' : {
      'epochs' : 50,
      'batch_size' : 32,
      'lr' : 1e-4,
      'weight_decay' : 0.01
    }
  }
}