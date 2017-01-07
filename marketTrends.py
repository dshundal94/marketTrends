import numpy as np 
import pandas as pd 
import csv as csv
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

#read in the various file names corresponding to different areas in the central valley
train_20701 = pd.read_csv('C:/Users/Damanjit/Documents/marketTrends/data/area20701.csv')
train_20702 = pd.read_csv('C:/Users/Damanjit/Documents/marketTrends/data/area20702.csv')
train_20703 = pd.read_csv('C:/Users/Damanjit/Documents/marketTrends/data/area20703.csv')
train_20704 = pd.read_csv('C:/Users/Damanjit/Documents/marketTrends/data/area20704.csv')
train_20705 = pd.read_csv('C:/Users/Damanjit/Documents/marketTrends/data/area20705.csv')
train_20706 = pd.read_csv('C:/Users/Damanjit/Documents/marketTrends/data/area20706.csv')
train_20707 = pd.read_csv('C:/Users/Damanjit/Documents/marketTrends/data/area20707.csv')
train_20708 = pd.read_csv('C:/Users/Damanjit/Documents/marketTrends/data/area20708.csv')
train_20801 = pd.read_csv('C:/Users/Damanjit/Documents/marketTrends/data/area20801.csv')
train_20802 = pd.read_csv('C:/Users/Damanjit/Documents/marketTrends/data/area20802.csv')
train_20803 = pd.read_csv('C:/Users/Damanjit/Documents/marketTrends/data/area20803.csv')
train_20806 = pd.read_csv('C:/Users/Damanjit/Documents/marketTrends/data/area20806.csv')
train_20901 = pd.read_csv('C:/Users/Damanjit/Documents/marketTrends/data/area20901.csv')

#This shows when a part of the code is processed
def status(code):

    print 'Processing',code,': ok'

#Need to create Days on Market for training data
def create_marketDays(Dframe):
	Dframe['Listing Date'] = pd.to_datetime(Dframe['Listing Date'])
	Dframe['Pending Date'] = pd.to_datetime(Dframe['Pending Date'])
	Dframe['Selling Date'] = pd.to_datetime(Dframe['Selling Date'])

	Dframe['Days on Market'] = Dframe['Pending Date'] - Dframe['Listing Date']
	Dframe['Days on Market'] = (Dframe['Days on Market'] / np.timedelta64(1, 'D')).astype(int)

	status('Days on Market')


training_list = [train_20705, train_20901]

#Create the Days on Market for the different files
for train in training_list:
	create_marketDays(train)

#Resample the data to have selling data as an index, grouped by Zip Code. 
def resample_data(training_data):
	#Days on Market
	resampled = training_data.set_index('Selling Date').groupby('Address - Zip Code').resample('6M').median()
	resampled.drop('Address - Zip Code', axis = 1, inplace = True)
	resampled.unstack(level = 0).plot(y = ['Days on Market'])
	plt.legend(loc="upper left", prop={'size':7}, bbox_to_anchor=(1,1))
	plt.ylabel('Days on Market')
	plt.title('Days on Market vs. Time')
	plt.show()

	#Listing Price and Selling Price
	resampled1 = training_data.set_index('Selling Date').groupby('Address - Zip Code').resample('2M').median()
	resampled1.drop('Address - Zip Code', axis = 1, inplace = True)
	resampled1.unstack(level = 0).plot(y = ['Listing Price', 'Selling Price'])
	plt.legend(loc = "upper left", prop={'size':7}, bbox_to_anchor=(1,1))
	plt.ylabel('Median Price')
	plt.title('Listing and Selling Median Price vs. Time')
	plt.show()

	#Price per square feet
	resampled2 = training_data.set_index('Selling Date').groupby('Address - Zip Code').resample('2M').median()
	resampled2.drop('Address - Zip Code', axis = 1, inplace = True)
	resampled2.unstack(level = 0).plot(y = ['Price Per Sq Ft'])
	plt.legend(loc = "upper left", prop={'size':7}, bbox_to_anchor=(1,1))
	plt.ylabel('Price Per Square Feet')
	plt.title('Price Per Sq Ft vs. Time')
	plt.show()

	status("Plots")

#Create the Plots for the two different biggest cities
for train1 in training_list:
	resample_data(train1)
	
#combine all the files for a larger more general dataset. It'll correspond to San Joaquin Area 
    
upd_training_list = [train_20701, train_20702, train_20703, train_20704, train_20706, train_20707, train_20708, train_20801, train_20802, train_20803, train_20806]

#create a columns of Days on Market for each of the areas of training data

for train in upd_training_list:
    create_marketDays(train)


combined1 = train_20901.append(train_20701)
combined2 = combined1.append(train_20702)
combined3 = combined2.append(train_20703)
combined4 = combined3.append(train_20704)
combined5 = combined4.append(train_20705)
combined6 = combined5.append(train_20706)
combined7 = combined6.append(train_20707)
combined8 = combined7.append(train_20708)
combined9 = combined8.append(train_20801)
combined10 = combined9.append(train_20802)
combined11 = combined10.append(train_20803)
combined12 = combined11.append(train_20806)
# Don't need the zip code
combined12.drop('Address - Zip Code', inplace = True, axis =1)

#only focusing on the price per square feet to forecast trends
combined_areas = combined12.set_index('Selling Date').resample('SM').median()
combined_areas.plot(y = ['Price Per Sq Ft'])
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.ylabel('Price Per Square Feet')
plt.title('San Joaquin Area: Price Per Sq Ft vs. Time')
plt.show()

#create timeseries with only Price Per Sq Ft
ts = combined_areas['Price Per Sq Ft']

#create a function to check whether the timeseries is stationary, implemented with help from Analytics Vidha
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12, center = False).mean()
    rolstd = timeseries.rolling(window=12, center = False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color = 'blue', label = 'Original')
    mean = plt.plot(rolmean, color='red', label = 'Rolling Mean')
    std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block = False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

#test out if series is stationary
test_stationarity(ts)

#reduce the non_stationarity of the series
ts_log = np.log(ts)
expweighted_avg = ts_log.ewm(span = 10, ignore_na = False, min_periods = 0, adjust = True).mean()
plt.plot(ts_log)
plt.plot(expweighted_avg, color='red')
plt.legend(loc = 'best')
plt.title('Rolling mean in Red vs Time')
plt.show()

#Take the difference to reduce trend
ts_log_ewma_diff = ts_log - expweighted_avg
test_stationarity(ts_log_ewma_diff)

#taking the difference
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)

#ACF and PACF plots: ACF - Autocorrelation Function, PACF - Partial ACF
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_ewma_diff, nlags=20)
lag_pacf = pacf(ts_log_ewma_diff, nlags=20, method='ols')
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_ewma_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_ewma_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_ewma_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_ewma_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

#forecast the timeseries using ARIMA
model = ARIMA(ts_log, order=(2, 1, 1))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues - ts_log_diff)**2))
plt.show()

#convert the fitted values back to original scale
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
plt.legend(loc = 'best')
plt.show()


