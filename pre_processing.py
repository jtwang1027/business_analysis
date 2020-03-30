import pandas as pd
import calendar
from datetime import datetime, date
# from weather_dict import wdict
import os

data=pd.read_csv(r'C:\Users\jwang\Desktop\quickly\items_transactions\items-2019-01-01-2020-02-15.csv')


# wdata= pd.read_excel(r'C:\Users\jwang\Desktop\quickly\weather_data.xlsx')
# cdata= pd.read_excel(r'C:\Users\jwang\Desktop\quickly\condition_data-april_2019.xlsx')

# wc= pd.merge(cdata,wdata, on='Date') # does intersection of data, not UNION
# wc['Date']=pd.to_datetime(wc.Date)


#imported as module
#wdict={'Partly Sunny':0, 'Rain':-2, 'Cloudy':-2, 'Mostly Sunny':1, 'Mostly Cloudy':-1,
#       'Partly Cloudy':0, 'Clear':2 , 'Hazy': -2, 'PM Thunderstorms':-3, 'AM Clouds/PM Sun': 1}

#removes soft/grand opening days
#datetime conversions
def preprocess(dat):
    '''
    returns sales per hour
    '''
    dat.dropna(subset=['Device Name'],inplace= True)
    dat=dat[dat['Qty']<15] #remove bulk orders
    #device name = nan is invoice/bulk orders
    dat['Date']=pd.to_datetime(dat.Date)
    dat=dat[dat.Date>'2018-12-01'] # remove grand opening and earlier dates


    def timesplit(t):
        '''returns time of day given in minutes'''
        (h, m, s) = t.split(':')
        return(int(h) * 60 + int(m) + int(s)/60)  
    
    dat['Ltime']=dat.Time.apply(timesplit)

    def hours(t):
        '''returns hour of the day'''
        (h, m, s) = t.split(':')
        return(int(h))
    dat['Hour']=dat.Time.apply(hours)
    
    
    def dollar(y):
        '''remove dollar sign by removing 1st element'''
        return([float(x[1:]) for x in y] )
    if '$' in dat['Gross Sales'][0]: #so it's not done more than once
        dat[['Gross Sales', 'Discounts', 'Net Sales', 'Tax']]=dat[['Gross Sales', 'Discounts', 'Net Sales', 'Tax']].apply(dollar)


    dat=dat[['Date','Hour','Gross Sales']]
    dat=dat.groupby(['Date','Hour'])['Gross Sales'].sum().reset_index()

    dat['Day'] = dat.Date.apply(lambda x: calendar.day_name[x.weekday()])
    dat = pd.get_dummies(dat, prefix=[None], columns=['Day'])

    #split date into month-date-year
    dat['Month']= dat.Date.dt.month
    dat['date']= dat.Date.dt.day
    dat['Year']= dat.Date.dt.year
    dat = dat.drop(['Date'], axis=1)

    return(dat)

data1=preprocess(data)
data1.to_csv(r'C:\Users\jwang\PycharmProjects\business_analysis\feb_2020-dataset.csv')





