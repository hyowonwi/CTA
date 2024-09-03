import sys
sys.path.append("..")
from configs.default_config import get_default_configs

def get_config():
  config = get_default_configs()
  
  # training
  training = config.training
  training.lr = 1e-4

  # data
  data = config.data
  data.dataset = 'AirQuality'
  data.dataset_path = './dataset/AirQuality_seqlen24_07masked'
  data.seq_len = 24
  data.feature_num = 132

  # model
  model = config.model
  model.model_type = "VAE_AE" # AE_AE, VAE_AE
  model.first_hidden_channels = 64
  model.first_latent_channels = 64
  model.first_num_layers = 3
  model.first_step_size = 1e-1

  model.second_hidden_channels = 32
  model.second_latent_channels = 64
  model.second_num_layers = 4
  model.second_step_size = 5e-1

  return config