params = {
  'RNN' : {
    'data' : {
        'input_length' : 12,
        'output_length' : 1
    },
    'model' : {
      'hidden_size' : 32,
      'num_layers' : 1          
    },
    'train' : {
      'epochs' : 50,
      'batch_size' : 32,
      'lr' : 1e-5,
    }
  },
  'LSTM' : {
    'data' : {
        'input_length' : 12,
        'output_length' : 1
    },
    'model' : {
      'hidden_size' : 32,
      'cell_size' : 32,
      'num_layers' : 1        
    },
    'train' : {
      'epochs' : 50,
      'batch_size' : 32,
      'lr' : 1e-5,
    }
  },
  'GRU' : {
    'data' : {
        'input_length' : 12,
        'output_length' : 1
    },
    'model' : {
      'hidden_size' : 32,
      'num_layers' : 1        
    },
    'train' : {
      'epochs' : 50,
      'batch_size' : 32,
      'lr' : 1e-5,
    }
  }
}