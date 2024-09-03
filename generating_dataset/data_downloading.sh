# Codes from SAITS
cd raw_data

# Stocks in raw_data from https://finance.yahoo.com/quote/GOOG/history?p=GOOG

# for Air-Quality
mkdir AirQuality && cd AirQuality
wget http://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip
unzip PRSA2017_Data_20130301-20170228.zip

# for Electricity 
cd .. && mkdir Electricity && cd Electricity
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip
unzip LD2011_2014.txt.zip


cd .. && mkdir Energy && cd Energy
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv
