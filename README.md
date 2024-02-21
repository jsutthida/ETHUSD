# ETHUSD price forecasting 

This project aims to study cryptocurentcy ETHUSD price. I will uses ARIMA and SARIMAX for building the forecasts model.

First, using yfinance to download historical ETHUSD price data.

```
import yfinance as yf
import pandas as pd

yf.pdr_override()
df = pdr.get_data_yahoo('ETH-USD', start = '2017-11-09', end = '2024-01-01')
```
Dataset:
```
     Date         Open	          High	        Low	        Close	      AdjClose	      Volume						
2017-11-09	308.644989	329.451996	307.056000	320.884003	320.884003	893249984	
2017-11-10	320.670990	324.717987	294.541992	299.252991	299.252991	885985984	
2017-11-11	298.585999	319.453003	298.191986	314.681000	314.681000	842300992	
2017-11-12	314.690002	319.153015	298.513000	307.907990	307.907990	1613479936	
2017-11-13	307.024994	328.415009	307.024994	316.716003	316.716003	1041889984	
...	...	...	...	...	...	...	...
2023-12-27	2231.393066	2392.608643	2215.140381	2378.739990	2378.739990	14161342927	
2023-12-28	2380.200684	2445.017578	2338.703857	2347.566162	2347.566162	15660799060	
2023-12-29	2346.843750	2386.004639	2262.975830	2300.690674	2300.690674	12536968996	
2023-12-30	2300.399658	2322.021484	2270.011963	2292.065430	2292.065430	6888195427	
2023-12-31	2291.945312	2318.512939	2261.394287	2281.471191	2281.471191	6871481744	
2244 rows × 7 columns
```

Next step would be split the data into 2 sets which is train dataset (80% of data) and test dataset (20% of data). So, I have 795 data samples in the train dataset.
```
#train 80% of data
data = df[:round(len(df)*80/100)]
test = df[round(len(df)*80/100)-1:]
```

![traindata](https://github.com/jsutthida/ETHUSD/assets/160230541/e182f12f-9ca6-492c-b2b7-46fb7947310c)
![testdata](https://github.com/jsutthida/ETHUSD/assets/160230541/07804253-6016-445c-9079-1dd28c9ab2f2)


### ARIMA

ARIMA (autoregressive integrated moving average) is a statistical model used for time series analysis and forecasting. 

Let's see the steps for using the ARIMA model with time series data:
1. Set the index to the date or time column.
2. Visualise the data and conduct the statistical tests.
3. Make data stationary by diffrencing if applicable.
4. Identify ARIMA model using ACF and PACF or using Auto ARIMA.
5. Estimate ARIMA model peremeter (p,d,q).
6. Choose most suitable ARIMA model.

Checking the data is stationary whether or not by using ADF (Augmented Dickey-Fuller) Test, if p-value < 0.05 then the data is stationary, if not the data must be diffrenced.

```
from statsmodels.tsa.stattools import adfuller

adf_res = adfuller(df['Close'], autolag = 'AIC')
print('p-Values:' + str(adf_res[1]))
```

p-Values is 0.5959299049792007

According to the ADF test, p-Values > 0.05 so I have to convert it into a stationary data by diffrencing.

```
data_diff=data['Close'].diff(periods=1)

adf_res = adfuller(data_diff.dropna(), autolag = 'AIC')
print('p-Values:' + str(adf_res[1]))
```

p-Values is 1.0248692915096733e-18

![data diff](https://github.com/jsutthida/ETHUSD/assets/160230541/c47da8a7-008c-41a2-9ce2-8324a0708158)

Stationary is checked! 
Next step, determining the order of the ARIMA model (p, d, q) based on the data's characteristics by GRID Search.

```
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings("ignore")

# Test order for finding the lowest AIC
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
aic = []
for param in pdq:
    try:
        model = ARIMA(data['Close'].dropna(),order = param).fit()
        print(f'Order: {param}')
        print(f'AIC: {model.aic:.4f}')
    except:
        continue
```
The lowest AIC is Order: (1, 2, 2)
AIC: 20805.8101

Then fit this order to ARIMA model.

```
arima=ARIMA(data['Close'],order=(1,2,2))
predicted=arima.fit().predict();predicted

arima1=ARIMA(test['Close'],order=(1,2,2))
predicted1=arima1.fit().predict();predicted1

plt.figure(figsize=(15,4))
plt.plot(data['Close'],color='green',label='Actual')
plt.plot(predicted,color='orange',label='Predicted')
plt.title('ETHUSD price history')
plt.legend()
```
![traindata predict](https://github.com/jsutthida/ETHUSD/assets/160230541/ce5ea0a7-a3a8-41e0-b434-b09581aa2891)

Training the ARIMA model, forecast the time series using return model.predict() and fit the model using model.fit().
Analysing the model's performance by MSE.

```
predicted1=arima1.fit().predict(start=futureDate.index[0],end=futureDate.index[-1]);predicted1
arima1.fit().predict(start=futureDate.index[0],end=futureDate.index[-1]).plot(color='orange')

plt.figure(figsize=(15,4))
plt.plot(test['Close'],color='green',label='Actual')
plt.plot(predicted1,color='orange',label='Predicted')
plt.title('ETHUSD price history')
plt.legend()
```

MSE : 80.220206202715

![arimaforecast](https://github.com/jsutthida/ETHUSD/assets/160230541/92fc05a2-13f8-49fa-953b-b52795fd101e)

![arimaforecastall](https://github.com/jsutthida/ETHUSD/assets/160230541/12be1b17-1aa6-4795-8fdc-e6534cee8411)

### SARIMAX

SARIMAX (Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors) is a generalization of the ARIMA model that considers both seasonality and exogenous variables. 
In this part I'll utilize auto_arima function to find the best parameters by setting quaterly seasonal.

```
import pmdarima as pmd

modelauto=pmd.auto_arima(data['Close'],start_p=1,start_q=1,test='adf',m=4,seasonal=True,trace=True)
```
Best model:  ARIMA(1,1,0)(2,0,2)[4]   

Training the SARIMAX model. Analysing the model's performance by MSE.

```
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima=SARIMAX(data['Close'],order=(1,1,0),seasonal_order=(2,0,2,4))
spredicted=sarima.fit().predict();spredicted

plt.figure(figsize=(15,4))
plt.plot(data['Close'],label='Actual')
plt.plot(spredicted,label='Predicted')
plt.legend()
```

![sarima traindata predict](https://github.com/jsutthida/ETHUSD/assets/160230541/876cff2f-6840-4494-9390-f614da9daa5e)

MSE : 79.79643391832631

```
sarima1=SARIMAX(test['Close'],order=(1,1,0),seasonal_order=(2,0,2,4))
spredicted1=sarima1.fit().predict();spredicted1

smodel.fit(test['Close'])
pred1=smodel.predict(n_periods=15);pred1
plt.plot(pred1, color='orange')
```

![sarimaforecast](https://github.com/jsutthida/ETHUSD/assets/160230541/199204bf-7e8c-4b62-80e2-f9284771338f)

![sarimaforecastall](https://github.com/jsutthida/ETHUSD/assets/160230541/3af69a98-c208-4a18-87ec-35281a57cc1f)

