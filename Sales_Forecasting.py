# Sales Forecasting Project
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=pd.read_csv('Walmart.csv')
df['Date']=pd.to_datetime(df['Date'],dayfirst=True)
df['Year']=df['Date'].dt.year
df['Month']=df['Date'].dt.month
df['Week']=df['Date'].dt.isocalendar().week.astype(int)

X=df[['Store','Holiday_Flag','Temperature','Fuel_Price','CPI','Unemployment','Year','Month','Week']]
y=df['Weekly_Sales']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestRegressor(n_estimators=200,random_state=42)
model.fit(X_train,y_train)
pred=model.predict(X_test)
print('MAE:',mean_absolute_error(y_test,pred))
print('RMSE:',np.sqrt(mean_squared_error(y_test,pred)))
print('R2 Score:',r2_score(y_test,pred))
