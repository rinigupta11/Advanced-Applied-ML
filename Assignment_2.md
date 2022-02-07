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
