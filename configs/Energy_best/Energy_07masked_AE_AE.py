import sys
sys.path.append("..")
from configs.default_config import get_default_configs

def get_config():
  config = get_default_configs()
  
  # training
  training = config.training
  training.lr = 5e-4

  # data
  data = config.data
  data.dataset = 'AirQuality'
  data.dataset_path = './dataset/Energy_seqlen48_07masked'
  data.seq_len = 48
  data.feature_num = 28

  # model
  model = config.model
  model.model_type = "AE_AE" # AE_AE, VAE_AE
  model.first_hidden_channels = 96
  model.first_latent_channels = 64
  model.first_num_layers = 4
  model.first_step_size = 5e-1

  model.second_hidden_channels = 96
  model.second_latent_channels = 64
  model.second_num_layers = 4
  model.second_step_size = 5e-1

  return config