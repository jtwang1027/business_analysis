'''
In building a model, we're interested in how weather affects sales.
We will use this script to scrape weather data from Durham, NC from March 2019 to present
'''

#add weather conditions into model: sunny/cloudy ,or visibility
#ie https://www.wunderground.com/history/daily/us/nc/raleigh-durham-airport/KRDU/date/2019-2-9


# Importing the necessary modules
from selenium import webdriver
import pandas as pd
from time import sleep

# creating CSV file to be used - each title is separated by a comma
#file = open(os.path.expanduser(r"~/Desktop/Weather Data.csv"), "wb")
#file.write(b"Date,Mean Temperature,Max Temperature,Min Temperature,Heating Degree Days, Dew Point, Average Humidity, Max Humidity, Minimum Humidity, Precipitation, Sea Level Pressure, Average Wind Speed, Maximum Wind Speed, Visibility, Events" + b"\n")
driver= webdriver.Chrome('C:/path/to/chromedriver')

wdata=pd.DataFrame()

#starting year, month
year='2019'
month='3'

#url format for a given date
murl='https://www.wunderground.com/history/monthly/us/nc/durham/KRDU/date/'+year+'-'+ month+'?cm_ven=localwx_history'

driver.get(murl) #go to URL
sleep(5) # wait for page to load

# TEMPERATURE
avtemp=[]
for tr in driver.find_elements_by_xpath('//*[@id="inner-content"]/div[2]/div[3]/div/div[1]/div/div/city-history-observation/div/div[2]/table/tbody/tr/td[2]/table/tbody/tr'):
#//*[@id="inner-content"]/div[2]/div[3]/div/div[1]/div/div/city-history-observation/div/div[2]/table/tbody/tr/td[1]/table/tbody/tr[2]/td
    tds= tr.find_elements_by_tag_name('td')[1] # [1] gives u second column, which is only averages
    avtemp.append(tds.text)

#PRECIPITATION
rain=[]
for tr in driver.find_elements_by_xpath('//*[@id="inner-content"]/div[2]/div[3]/div/div[1]/div/div/city-history-observation/div/div[2]/table/tbody/tr/td[7]/table/tbody/tr'):
    tds= tr.find_elements_by_tag_name('td')[1] # [1] gives u second column, which is only averages
    rain.append(tds.text)
#dates
dates=[]
for tr in  driver.find_elements_by_xpath('//*[@id="inner-content"]/div[2]/div[3]/div/div[1]/div/div/city-history-observation/div/div[2]/table/tbody/tr/td[1]/table/tbody/tr'):
    tds= tr.find_elements_by_tag_name('td')
    dates+= [td.text for td in tds]

codate= [month+'/'+ x+'/'+ year for x in dates]


wdf2= pd.DataFrame( list(zip(codate[1:], avtemp[1:], rain[1:])), columns= ['Date', 'Temp', 'Rain'] )
wdata=pd.concat([wdata,wdf2])


wdata.to_excel(r'C:\path\to\data.xlsx', index= False)

driver.quit()


