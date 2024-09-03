# Codes from SAITS

#####################################
# generate UCI Beijing air quality dataset
#####################################

python gene_UCI_BeijingAirQuality_dataset.py \
  --file_path raw_data/AirQuality/PRSA_Data_20130301-20170228 \
  --seq_len 24 \
  --artificial_missing_rate 0.3 \
  --dataset_name ../dataset/AirQuality_seqlen24_03masked \
  --saving_path ../dataset

# python gene_UCI_BeijingAirQuality_dataset.py \
#   --file_path raw_data/AirQuality/PRSA_Data_20130301-20170228 \
#   --seq_len 24 \
#   --artificial_missing_rate 0.5 \
#   --dataset_name ../dataset/AirQuality_seqlen24_05masked \
#   --saving_path ../dataset

# python gene_UCI_BeijingAirQuality_dataset.py \
#   --file_path raw_data/AirQuality/PRSA_Data_20130301-20170228 \
#   --seq_len 24 \
#   --artificial_missing_rate 0.7 \
#   --dataset_name ../dataset/AirQuality_seqlen24_07masked \
#   --saving_path ../dataset

# #####################################
# # generate GoogleStock dataset
# #####################################

# python gene_GoogleStock_dataset.py \
#     --file_path raw_data/GoogleStock/GOOGL.csv \
#     --artificial_missing_rate 0.3 \
#     --seq_len 20 \
#     --dataset_name ../dataset/GoogleStock_seqlen20_03masked \
#     --saving_path ../dataset

# python gene_GoogleStock_dataset.py \
#     --file_path raw_data/GoogleStock/GOOGL.csv \
#     --artificial_missing_rate 0.5 \
#     --seq_len 20 \
#     --dataset_name ../dataset/GoogleStock_seqlen20_05masked \
#     --saving_path ../dataset

# python gene_GoogleStock_dataset.py \
#     --file_path raw_data/GoogleStock/GOOGL.csv \
#     --artificial_missing_rate 0.7 \
#     --seq_len 20 \
#     --dataset_name ../dataset/GoogleStock_seqlen20_07masked \
#     --saving_path ../dataset


# #####################################
# # generate UCI Electricity dataset
# #####################################

# python gene_UCI_electricity_dataset.py \
#   --file_path raw_data/Electricity/LD2011_2014.txt \
#   --artificial_missing_rate 0.3 \
#   --seq_len 48 \
#   --dataset_name ../dataset/Electricity_seqlen48_03masked\
#   --saving_path ../dataset

# python gene_UCI_electricity_dataset.py \
#   --file_path raw_data/Electricity/LD2011_2014.txt \
#   --artificial_missing_rate 0.5 \
#   --seq_len 48 \
#   --dataset_name ../dataset/Electricity_seqlen48_05masked \
#   --saving_path ../dataset

# python gene_UCI_electricity_dataset.py \
#   --file_path raw_data/Electricity/LD2011_2014.txt \
#   --artificial_missing_rate 0.7 \
#   --seq_len 48 \
#   --dataset_name ../dataset/Electricity_seqlen48_07masked \
#   --saving_path ../dataset


# # ####################################
# # generate UCI Energy dataset
# # ####################################

# python gene_UCI_Energy_dataset.py \
#   --file_path raw_data/Energy/energydata_complete.csv \
#   --artificial_missing_rate 0.3 \
#   --seq_len 48 \
#   --dataset_name ../dataset/Energy_seqlen48_03masked \
#   --saving_path ../dataset \

# python gene_UCI_Energy_dataset.py \
#   --file_path raw_data/Energy/energydata_complete.csv \
#   --artificial_missing_rate 0.5 \
#   --seq_len 48 \
#   --dataset_name ../dataset/Energy_seqlen48_05masked \
#   --saving_path ../dataset \

# python gene_UCI_Energy_dataset.py \
#   --file_path raw_data/Energy/energydata_complete.csv \
#   --artificial_missing_rate 0.7 \
#   --seq_len 48 \
#   --dataset_name ../dataset/Energy_seqlen48_07masked \
#   --saving_path ../dataset \
