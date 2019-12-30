#some names are missing still
#include secret menu?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
#from collections import Counter

filename=r'\path\items-2019-09-02.csv'

items= pd.read_csv(filename)
items['Date']=pd.to_datetime(items['Date'])
#exclude grand opening
items= items[items.Date>'2018-12-01'] 


drink_list=['Slush', 'Milk Tea', 'Tea', 'Snow','Hot Drinks', 'Coffee', 'Spring Drinks', 'Secret Menu']
#add in outlier removal/ large orders


#2 ways of identifying customers: ID and Name, customer ID maybe stored regardless of drink name used?
drink_orders=items[items.Category.isin(drink_list)]# subset of orders that are drinks

# convert from string w/ $ to float


drink_orders['Net Sales']=drink_orders['Net Sales'].apply(lambda x: float(x.strip('$'))) 

#subset that received discounts, Net Sales should be 0
disc=drink_orders[drink_orders['Net Sales']==0]
disc_freq= pd.value_counts(disc['Customer Name'])
disc_freq=pd.Series(disc_freq)

freq = pd.value_counts(drink_orders['Customer Name']) #total number drinks per individual
freq=pd.Series(freq)

final= pd.concat([freq,disc_freq], axis=1)
final.columns=['Purchased','Discounted']
final=final.fillna(0)
#final=final.sort_values(by=['Discounted'])

final['Expected Free']= final['Purchased'].apply(lambda x: np.floor(x/10) )
final['Mismatch']= final['Discounted']-final['Expected Free']

recent=[] # adding most recent date of discount
for x, row in final.iterrows(): 
    print(x)
    recent.append(pd.Timestamp.to_pydatetime(disc[disc['Customer Name']==x].Date.max()).strftime('%m/%d/%Y'))

final['Most Recent Disc']=recent


#cut off mismatches<=0 , they missed out on free drinks or weren't elgible yet
final= final[final['Mismatch']>0]

final=final.sort_values(by=['Mismatch'], ascending=False)
final.to_excel('190901-discounts.xlsx')



#number of discounts vs time 

drink_orders['wk_st']=drink_orders['Date']- pd.to_timedelta(drink_orders['Date'].dt.weekday,unit='D') #Mon week starts

drink_orders['Discounts']=drink_orders['Discounts'].apply(lambda x: float(x.strip('$')))  #discounted convert to float
dr_disc= drink_orders[drink_orders['Discounts']<0] # sum drinks have discounts, but not full discounts?
drink_orders['Gross Sales']=drink_orders['Gross Sales'].apply(lambda x: float(x.strip('$'))) 
#as percentage of gross sales
weekly= (drink_orders.groupby(['wk_st'])['Discounts'].sum() / drink_orders.groupby(['wk_st'])['Gross Sales'].sum())*-100

#plots discounts on drinks % per week
plt.scatter(weekly.index[1:], weekly[1:])
plt.ylabel('Percentage Discount on Drinks')
plt.xlabel('Week')
plt.title('Drink discount / Drink gross sales by week')
plt.show()





