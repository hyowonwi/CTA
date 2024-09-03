import sys
sys.path.append("..")
from configs.default_config import get_default_configs

def get_config():
  config = get_default_configs()
  
  # training
  training = config.training
  training.lr = 1e-3
  training.batch_size = 64
  # data
  data = config.data
  data.dataset = 'Stocks'
  data.dataset_path = './dataset/GoogleStock_seqlen20_03masked'
  data.seq_len = 20
  data.feature_num = 6

  # model
  model = config.model
  model.model_type = "AE_AE" # AE_AE, VAE_AE
  model.first_hidden_channels = 16
  model.first_latent_channels = 16
  model.first_num_layers = 5
  model.first_step_size = 1e-1

  model.second_hidden_channels = 16
  model.second_latent_channels = 16
  model.second_num_layers = 5
  model.second_step_size = 1e-1

  return config