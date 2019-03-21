import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from my_timeseries2 import predict_time_series

#input data
data=pd.read_csv('14.csv')
timestep=15
nmsph=int(60/timestep)
print(data.head(20))
num_weeks=3 #Duration of prediction (number of weeks)
num_weeks_plot=12 #number of last weeks to plot
#rename columns
data.rename(columns={'dttm_utc':'date_time', 'value':'total_calls'}, inplace=True)

predictdata=predict_time_series(data,num_weeks)

#renaming the columns
predictdata.rename(columns={'total_calls':'Consumption'}, inplace=True)
data.rename(columns={'total_calls':'Consumption'}, inplace=True)

predictdata.plot(x='date_time', y='Consumption')
w_data=pd.concat([data,predictdata])
w_data.iloc[-nmsph*24*7*num_weeks_plot:].plot(x='date_time', y='Consumption')

plt.show()
