import ml_collections


def get_default_configs():
  config = ml_collections.ConfigDict()

  # model
  config.model = model = ml_collections.ConfigDict()
  model.saving_path = './test/etc'
  model.checkpoint_path = '.'

  model.model_type = "AE_AE" # VAE, AE, AE_AE, VAE_AE
  model.first_hidden_channels = 0
  model.first_latent_channels = 0
  model.first_num_layers = 0
  model.first_step_size = 0.0

  model.second_hidden_channels = 0
  model.second_latent_channels = 0
  model.second_num_layers = 0
  model.second_step_size = 0.0

  
  # training
  config.training = training = ml_collections.ConfigDict()

  training.batch_size = 128
  training.epoch = 1000
  training.patience = 30
  training.num_workers = 4

  training.lr = 0.001
  training.atol = 1e-9
  training.rtol = 1e-7
  training.solver = 'rk4'

  training.weight_decay = 1e-4
  training.adjoint = 1
  training.kld_weight = 1e-4
  training.masked_imputation_task = 1

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset_path = '.'
  data.seq_len = 0
  data.feature_num = 0
  data.interpolation_type = 'cubic'
  
  return config