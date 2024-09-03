# r_missing=70% VAE_AE
python main.py --config ./configs/Energy_best/Energy_07masked_VAE_AE.py --config.model.saving_path ./experiments/Energy_07masked_VAE_AE_reproduce --mode train --log_path ./experiments/Energy_07masked_VAE_AE_reproduce/train.log
# r_missing=70% AE_AE
python main.py --config ./configs/Energy_best/Energy_07masked_AE_AE.py --config.model.saving_path ./experiments/Energy_07masked_AE_AE_reproduce  --mode train --log_path ./experiments/Energy_07masked_AE_AE_reproduce/train.log