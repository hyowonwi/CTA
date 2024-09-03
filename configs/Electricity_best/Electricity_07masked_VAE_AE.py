import sys
sys.path.append("..")
from configs.default_config import get_default_configs

def get_config():
  config = get_default_configs()
  # data
  data = config.data
  data.dataset = 'Electricity'
  data.dataset_path = './dataset/Electricity_seqlen48_07masked'
  data.seq_len = 48
  data.feature_num = 370
  
  # training
  training = config.training
  training.lr = 1e-3

  # model
  model = config.model
  model.model_type = "VAE_AE" # AE_AE, VAE_AE

  model.first_step_size = 1e-1
  model.first_hidden_channels = 128
  model.first_num_layers = 5
  model.first_latent_channels = 512

  model.second_step_size = 5e-1
  model.second_hidden_channels = 256
  model.second_num_layers = 4
  model.second_latent_channels = 128

  return config