# Locally Weighted Linear Regression versus Random Forest Regression
By Rini Gupta and Kimya Shirazi 

We will be using a simple cars dataset with one input feature on both regression methods. In order to compare performance, we include a cross-validation step
comparing the mean-squared error between the two methods. 

#### Import Necessary Libraries 
```
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import train_test_split as tts
from scipy import linalg
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
```
When we first load the data in, we visually inspected the car weight versus miles per gallon using a scatter plot. This was the result: 

![image](https://user-images.githubusercontent.com/76021844/152820740-ae918b4e-e77c-4d2f-8ccb-21fe9207cd47.png)

We then selected the input feature (in both the form of an array and as a dataframe) as well as the target data (MPG).  

```
data = pd.read_csv('cars.csv')
x = data['WGT'].values # array
X = data['WGT'] # dataframe
y = data['MPG'].values # target
```

## Locally Weighted Linear Regression
Next, we will examine the performance of a locally weighted linear regression model. 
```
# Tricubic Kernel
def tricubic(x):
  return np.where(np.abs(x)>1,0,70/81*(1-np.abs(x)**3)**3)

# Epanechnikov Kernel
def Epanechnikov(x):
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2)) 

# Quartic Kernel
def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2) 
```

The function below is the implementation we used for the locally weighted linear regression:
```
def lowess_reg(x, y, xnew, kern, tau):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    # IMPORTANT: we expect x to the sorted increasingly
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        #theta = linalg.solve(A, b) # A*theta = b
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 
    f = interp1d(x.ravel(), yest, fill_value='extrapolate')
    return f(xnew)
```

For our analysis of locally weighted linear regression, we will experiment with all three of the kernels to see which performs the best. 

```
# Tricubic kernel 
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=2021)

acc_score = []

for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:].values,X.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    scale = ss()
    xtrain_scaled = scale.fit_transform(X_train.reshape(-1,1))
    xtest_scaled = scale.transform(X_test.reshape(-1,1))
    pred_values = lowess_reg(xtrain_scaled.ravel(), y_train, xtest_scaled.ravel(), tricubic, .4)        
    acc_score.append(MSE(pred_values, y_test))
avg_acc_score = sum(acc_score)/k

print('MSE of each fold :  {}'.format(acc_score))
print('Avg MSE : {}'.format(avg_acc_score))

## Referenced https://www.askpython.com/python/examples/k-fold-cross-validation
```
The results of the tricubic kernel lowess model were: 
MSE of each fold :  [11.332985107185639, 19.11239609934669, 17.36674384258766, 23.689698736174996, 15.240672370371039, 13.869259425898996, 12.74639826110358, 17.634333510577488, 21.64516686537066, 24.356700203630663]
Avg MSE : 17.699435442224743

```
# Epanechnikov kernel 
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=2021)

acc_score = []

for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:].values,X.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    scale = ss()
    xtrain_scaled = scale.fit_transform(X_train.reshape(-1,1))
    xtest_scaled = scale.transform(X_test.reshape(-1,1))
    pred_values = lowess_reg(xtrain_scaled.ravel(), y_train, xtest_scaled.ravel(), Epanechnikov, .6)        
    acc_score.append(MSE(pred_values, y_test))
avg_acc_score = sum(acc_score)/k
print('MSE of each fold :  {}'.format(acc_score))
print('Avg MSE : {}'.format(avg_acc_score))
print('----------------------------------------------')

## Referenced https://www.askpython.com/python/examples/k-fold-cross-validation
```

The results of the Epanechnikov kernel lowess model were:
MSE of each fold :  [11.50068281570299, 19.229746106256346, 16.713493203307657, 23.283726256432914, 15.68693296194241, 14.077705974312815, 12.88041838397464, 17.664220522160583, 21.5220860387092, 24.035266642493177]
Avg MSE : 17.659427890529276

```
# Quartic kernel 
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=2021)

acc_score = []

for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:].values,X.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    scale = ss()
    xtrain_scaled = scale.fit_transform(X_train.reshape(-1,1))
    xtest_scaled = scale.transform(X_test.reshape(-1,1))
    pred_values = lowess_reg(xtrain_scaled.ravel(), y_train, xtest_scaled.ravel(), Quartic, .6)        
    acc_score.append(MSE(pred_values, y_test))
avg_acc_score = sum(acc_score)/k

print('MSE of each fold :  {}'.format(acc_score))
print('Avg MSE : {}'.format(avg_acc_score))

## Referenced https://www.askpython.com/python/examples/k-fold-cross-validation
```

The results of the quartic kernel lowess model were: 
MSE of each fold :  [11.437277992038139, 19.15051846263025, 16.86320518637241, 23.44167278802619, 15.48092421868573, 13.936508981385979, 13.056386569288353, 17.695286840208762, 21.55193570023347, 24.076182165664257]
Avg MSE : 17.668989890453354

### Hyperparameter Optimization

For locally-weighted linear regression, we can try to optimize tau to obtain the best results. Out of the three different kernels, the Epanechnikov kernel yieled the lowest MSE value. As a result, for our hyperparameter optimization, we will continue to use the Epanechnikov kernel. 

```
# Epanechnikov kernel 
tau_results = []
for tau in np.linspace(.01, 1, 25): 
  k = 10
  kf = KFold(n_splits=k, shuffle=True, random_state=2021)

  acc_score = []

  for train_index , test_index in kf.split(X):
      X_train , X_test = X.iloc[train_index,:].values,X.iloc[test_index,:].values
      y_train, y_test = y[train_index], y[test_index]
      scale = ss()
      xtrain_scaled = scale.fit_transform(X_train.reshape(-1,1))
      xtest_scaled = scale.transform(X_test.reshape(-1,1))
      pred_values = lowess_reg(xtrain_scaled.ravel(), y_train, xtest_scaled.ravel(), Epanechnikov, tau)        
      acc_score.append(MSE(pred_values, y_test))
  avg_acc_score = sum(acc_score)/k
  tau_results.append(avg_acc_score)
## Referenced https://www.askpython.com/python/examples/k-fold-cross-validation
```

We then plotted the resulting MSE values per each value of tau. 
![image](https://user-images.githubusercontent.com/76021844/152909021-854dca84-112e-4ab3-8859-76d52ae31610.png)

From there, we extracted the exact optimal value of tau.
```
optimal_tau = np.linspace(.01, 2, 50)[np.argmin(tau_results)]
optimal_tau
```

Then, we ran the model again with the optimal kernel and optimal value of tau for this dataset to get an average MSE value to compare with the random forest regressor. 

```
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=2021)

acc_score = []

for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:].values,X.iloc[test_index,:].values
    y_train, y_test = y[train_index], y[test_index]
    scale = ss()
    xtrain_scaled = scale.fit_transform(X_train.reshape(-1,1))
    xtest_scaled = scale.transform(X_test.reshape(-1,1))
    pred_values = lowess_reg(xtrain_scaled.ravel(), y_train, xtest_scaled.ravel(), Epanechnikov, optimal_tau)        
    acc_score.append(MSE(pred_values, y_test))
avg_acc_score = sum(acc_score)/k
tau_results.append(avg_acc_score)
print('MSE of each fold :  {}'.format(acc_score))
print('Avg MSE : {}'.format(avg_acc_score))
```
The final output of this was:
MSE of each fold :  [11.541070792970553, 19.295776420950126, 16.677534852169654, 23.188253372530607, 15.809113819696282, 14.116661765872486, 12.6881410725552, 17.67105448817393, 21.53164020038221, 24.006581540152368]
Avg MSE : 17.65258283254534

## Random Forest Regressor

Next, we examined the performance of a random forest regressor 



