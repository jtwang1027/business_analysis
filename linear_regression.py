import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt

data1=pd.read_csv(r'C:\Users\jwang\PycharmProjects\business_analysis\feb_2020-dataset.csv')
y=data1['Gross Sales']
X= data1.drop(['Gross Sales'],axis=1)
X=X.reindex(sorted(X.columns),axis=1) #put in alphabetical order

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size = 0.25, random_state = 42)

reg=linear_model.LinearRegression()
reg.fit(xtrain, ytrain)

y_pred=reg.predict(xtest)

mse=sqrt(mean_squared_error(ytest, y_pred))
f'RMSE: {mse}'



