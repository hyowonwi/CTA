# r_missing=30% VAE_AE
python main.py --config ./configs/Stocks_best/Stocks_03masked_VAE_AE.py --config.model.saving_path ./experiments/Stocks_03masked_VAE_AE_reproduce --mode train --log_path ./experiments/Stocks_03masked_VAE_AE_reproduce/train.log
# r_missing=30% AE_AE
python main.py --config ./configs/Stocks_best/Stocks_03masked_AE_AE.py --config.model.saving_path ./experiments/Stocks_03masked_AE_AE_reproduce  --mode train --log_path ./experiments/Stocks_03masked_AE_AE_reproduce/train.log

# r_missing=50% VAE_AE
python main.py --config ./configs/Stocks_best/Stocks_05masked_VAE_AE.py --config.model.saving_path ./experiments/Stocks_05masked_VAE_AE_reproduce --mode train --log_path ./experiments/Stocks_05masked_VAE_AE_reproduce/train.log
# r_missing=50% AE_AE
python main.py --config ./configs/Stocks_best/Stocks_05masked_AE_AE.py --config.model.saving_path ./experiments/Stocks_05masked_AE_AE_reproduce  --mode train --log_path ./experiments/Stocks_05masked_AE_AE_reproduce/train.log

# r_missing=70% VAE_AE
python main.py --config ./configs/Stocks_best/Stocks_07masked_VAE_AE.py --config.model.saving_path ./experiments/Stocks_07masked_VAE_AE_reproduce --mode train --log_path ./experiments/Stocks_07masked_VAE_AE_reproduce/train.log
# r_missing=70% AE_AE
python main.py --config ./configs/Stocks_best/Stocks_07masked_AE_AE.py --config.model.saving_path ./experiments/Stocks_07masked_AE_AE_reproduce  --mode train --log_path ./experiments/Stocks_07masked_AE_AE_reproduce/train.log