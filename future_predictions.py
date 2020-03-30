import pandas as pd
import xgboost as xgb
import pickle
import datetime
import itertools



xgb_model=xgb.Booster()
xgb_model.load_model(r'C:\Users\jwang\PycharmProjects\business_analysis\xgb-200215.bin')


'''
Input X columns (date= date within month):
['Hour', 'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
       'Wednesday', 'Month', 'date', 'Year']
'''


#prediction range
start= datetime.datetime.today()
numdays=14 #2 week prediction

#generate list of dates
date_list = [start + datetime.timedelta(days=x) for x in range(numdays)]
Xfut=pd.DataFrame({'Date':date_list})

#get associated weekdays
Xfut['Day'] = Xfut.Date.apply(lambda x: calendar.day_name[x.weekday()])
Xfut = pd.get_dummies(Xfut, prefix=[None], columns=['Day'])

# split date into month-date-year
Xfut['Month'] = Xfut.Date.dt.month
Xfut['date'] = Xfut.Date.dt.day
Xfut['Year'] = Xfut.Date.dt.year

Xfut=pd.concat([Xfut]*13) #expand for each open hour
Xfut=Xfut.sort_values(by=['Date'])

#hours open 11:30am - 11pm; or 11-23 hr
hours=list(itertools.chain.from_iterable(itertools.repeat(range(11,24), numdays)))
Xfut['Hour']=hours
Xfut = Xfut.drop(['Date'], axis=1)


Xfut=Xfut.reindex(sorted(X.columns),axis=1) #put in alphabetical order
pred=xgb_model.predict(Xfut)

Xfut['Gross Sales pred']=pred

Xfut.to_csv('C:/Users/jwang/PycharmProjects/business_analysis/predictions/feb_predictions-xgb_model.csv')
