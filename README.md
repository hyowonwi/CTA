
# Official Code for "Continuous-time Autoencoders for Regular and Irregular Time Series Imputation" Paper in WSDM'24
## install conda environments 

```
conda create -n cta python=3.9.7
conda activate cta
sh create_env.sh
```

```
cd generating_dataset
sh data_downloading.sh
cd ..
sh dataset_generating.sh
```


## train CTA for STOCKS 70%

### VAE-AE
```
python main.py --config ./configs/Stocks_best/Stocks_07masked_VAE_AE.py --config.model.saving_path ./experiments/Stocks_07masked_VAE_AE_reproduce --mode train --log_path ./experiments/Stocks_07masked_VAE_AE_reproduce/train.log
```

### AE-AE
```
python main.py --config ./configs/Stocks_best/Stocks_07masked_AE_AE.py --config.model.saving_path ./experiments/Stocks_07masked_AE_AE_reproduce  --mode train --log_path ./experiments/Stocks_07masked_AE_AE_reproduce/train.log
```

For other datasets, change `dataset name` and 'missing rate`

## test pretrained model for STOCKS 70%

### VAE-AE

```
python main.py --config ./configs/Stocks_best/Stocks_07masked_VAE_AE.py --config.model.saving_path ./experiments/Stocks_07masked_VAE_AE_pretrained --mode test --log_path ./experiments/Stocks_07masked_VAE_AE_pretrained/test.log
```

### AE-AE
```
python main.py --config ./configs/Stocks_best/Stocks_07masked_AE_AE.py  --config.model.saving_path ./experiments/Stocks_07masked_AE_AE_pretrained --mode test --log_path ./experiments/Stocks_07masked_AE_AE_pretrained/test.log
```
