import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import xgboost as xgb

data1=pd.read_csv(r'C:\Users\jwang\PycharmProjects\business_analysis\feb_2020-dataset.csv')


y=data1['Gross Sales']
X= data1.drop(['Gross Sales'],axis=1)
X=X.reindex(sorted(X.columns),axis=1) #put in alphabetical order

#feature_list=list(features.columns)
#RF

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size = 0.25, random_state = 42)


xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

xgb_model.fit(xtrain, ytrain)

y_pred = xgb_model.predict(xtest)

mse=sqrt(mean_squared_error(ytest, y_pred))
f'RMSE: {mse}'

#add variable importance
#add visualizations
#save model

#retrain on all data and save
xgb_model.fit(X,y)
xgb_model.save_model(r'C:\Users\jwang\PycharmProjects\business_analysis\xgb-200215.model')

xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
import matplotlib.pyplot as plt

xgb.plot_tree(xgb_model,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()


'''
Input X columns (date= date within month):
['Hour', 'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
       'Wednesday', 'Month', 'date', 'Year']
'''