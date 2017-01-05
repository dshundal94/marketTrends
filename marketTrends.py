import numpy as np 
import pandas as pd 
import csv as csv
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt

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


training_list = [train_20701, train_20702, train_20703, train_20704, train_20705, train_20706, train_20707, train_20708, train_20801, train_20802, train_20803, train_20806, train_20901]

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

#Create the Plots for the different files
for train1 in training_list:
	resample_data(train1)
	
## Some of the files are not outputting plots, because the files need to be cleaned 
## and filtered. SOme other files don't have enough data points to create general
## trends. These files should be excluded. 