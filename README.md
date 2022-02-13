# Locally Weighted Linear Regression versus Random Forest Regression
By Rini Gupta and Kimya Shirazi 

We will be using a simple cars dataset with one input feature on both regression methods. In order to compare performance, we include a cross-validation step
comparing the mean-squared error between the two methods. Cross-validation is a statistical method of evaluating and comparing learning algorithms by dividing data into two segments: one used to learn or train a model and the other used to validate the model. Additionally, the mean-squared error is a simple and common loss function that is helpful to compare the results of two regression algorithms. 

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
Next, we will examine the performance of a locally weighted linear regression model. Linear regression is a supervised learning algorithm used for computing linear relationships between input (X) and output (Y). In the instance of a non-linear relationship between X and Y, locally weighted linear regression is used. Locally weighted linear regression is a non-parametric algorithm, that is, the model does not learn a fixed set of parameters as is done in ordinary linear regression.  Rather parameters (tau) are computed individually for each query point x. While computing tau, a higher “preference” is given to the points in the training set lying in the vicinity of x than the points lying far away from x. Locally weighted linear regression includes numerous regression methods in a k-nearest neighbor meta-model. Furthermore, locally weighted linear regression is a memory-based approach to learning. It is called a "lazy learner" because it does not train until a query is posed to answer regarding prediction. 

Source: https://scholar.google.com/scholar_url?url=http://www.qou.edu/ar/sciResearch/pdf/distanceLearning/locallyWeighted.pdf&hl=en&sa=X&ei=LAUJYuH1MN6Sy9YPo_qOwA8&scisig=AAGBfm2QpPq7HtXB0QEbnMhW8fb43AQSmQ&oi=scholarr

https://xavierbourretsicotte.github.io/loess.html

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

For our analysis of locally weighted linear regression, we will experiment with all three kernels learned in class to analyze which performs the best. In locally weighted linear regression, both the training data and parameters are needed to make a prediction, as it is key to understand which points are close to the test point. The weight term is a function of the test point and the training data points, meaning it measures how close the test point is to each of the training data points. Such a distance measure is called a kernel function. Kernel functions will be useful in other learning algorithms as well, particularly in Support Vector Machines. For this project, the tricubic, quartic, and Epanechnikov kernel are used below as smoothing functions.

Source: https://www.jstor.org/stable/2245737?seq=2#metadata_info_tab_contents

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

For locally-weighted linear regression, we can try to optimize tau to obtain the best results. Out of the three different kernels, the Epanechnikov kernel yieled the lowest MSE value. As a result, for our hyperparameter optimization, we will continue to use the Epanechnikov kernel. Hyperparameter optimization in machine learning intends to find the hyperparameters of a given machine learning algorithm that deliver the best performance as measured on a validation set. Hyperparameters, in contrast to model parameters, are set by the machine learning engineer before training.

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

![image](https://user-images.githubusercontent.com/76021844/153276801-d8b78d3e-c690-4a43-a12f-7fc1b49da94b.png)

## Random Forest Regressor

Next, we examined the performance of a random forest regressor. In order to understand how the random forest regressor works, we first introduce the concept of a decision tree. A decision tree is actually quite a simple tree-like structure which trains on some labeleled data to then make predictions about new data. This process occurs by forming a hierarchy of decisions to make in the training process. The process of separating the different levels of the tree happens recursively, separating into homogenous groups (nodes) down to terminal nodes (Gromping 2009). 
![image](https://user-images.githubusercontent.com/76021844/153754695-8e7d0a5c-fbec-4b84-b90c-fb756e6696fd.png)

The random forest regressor model is an ensemble model that incorporates many decision trees into its structure to make a final prediction on data. Unlike an ordinary linear regressor, random forests can fit to accomodate non-linearities in the dataset. As a result, similar to Lowess, random forests are non-parametric (Gromping 2009). Random forests are advantageous over decision trees because they are better at preventing overfitting due to the ensemble nature of the model (incorporating several predictions). The individual decision trees within the forest are, as the name suggests, quite random and yield differing predictions. The random forest algorithm takes the average of each individual decision tree to make final predictions (Gromping 2019). Additionally, random forests group weak learners together to form stronger learners (boosting), another theoretical strength of the model. Random forests are regarded by data scientists as one of the "best performing learning algorithms" (Schonlau 2020). First, we ran the model with some hardcoded hyperparameters to get a rough idea of model performance. 

<img width="867" alt="image" src="https://user-images.githubusercontent.com/76021844/153780621-864777ac-93fc-48ee-83c1-d180ad89623f.png">


Source: https://towardsdatascience.com/a-quick-and-dirty-guide-to-random-forest-regression-52ca0af157f8 

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2701298/ (Peer Reviewed)

https://towardsdatascience.com/what-is-a-decision-tree-22975f00f3e1

https://www.tandfonline.com/doi/abs/10.1198/tast.2009.08199
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
    rf = RandomForestRegressor(n_estimators=100,max_depth=3)
    rf.fit(xtrain_scaled,y_train)
    pred_values = rf.predict(xtest_scaled)    
    acc_score.append(MSE(pred_values, y_test))
avg_acc_score = sum(acc_score)/k

print('MSE of each fold:  {}'.format(acc_score))
print('Avg MSE: {}'.format(avg_acc_score))

## Referenced https://www.askpython.com/python/examples/k-fold-cross-validation
```

The output of this block of code was:
MSE of each fold:  [10.519229346856164, 16.949036884302654, 18.930417456552682, 23.847069406609652, 15.755929545760456, 13.959282952567856, 12.680582125310195, 17.404585471427826, 20.771358047591054, 26.680258272157243]
Avg MSE: 17.74977495091358


Then, we sought to optimize the performance of this model by fine-tuning the hyperparameters. Specifically, we tested different values for n-estimators and max depth. N-estimators is the number of trees in the random forest and max depth is the maximum depth of each tree. 

```
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=2021)
hyper_values = []
for estimators in [100, 500, 1000]: 
  for depth in [1, 2, 3, 4, 5]: 
    acc_score = []

    for train_index , test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:].values,X.iloc[test_index,:].values
        y_train, y_test = y[train_index], y[test_index]
        scale = ss()
        xtrain_scaled = scale.fit_transform(X_train.reshape(-1,1))
        xtest_scaled = scale.transform(X_test.reshape(-1,1))
        rf = RandomForestRegressor(n_estimators=estimators,max_depth=depth)
        rf.fit(xtrain_scaled,y_train)
        pred_values = rf.predict(xtest_scaled)    
        acc_score.append(MSE(pred_values, y_test))
    hyper_values.append([estimators, depth, avg_acc_score])
    avg_acc_score = sum(acc_score)/k

## Referenced https://www.askpython.com/python/examples/k-fold-cross-validation
```
Then, the minimum MSE value was extracted from the hyper_values array to determine the optimal combination of hyperparameters. 

```
np.argmin(np.array(hyper_values)[:, 2])
hyper_values[13]
```

The results of this were 1000 estimators and a max-depth of 3. 

Next, we reran the model after that with the optimal hyperparameters: 
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
    rf = RandomForestRegressor(n_estimators=1000,max_depth=3)
    rf.fit(xtrain_scaled,y_train)
    pred_values = rf.predict(xtest_scaled)    
    acc_score.append(MSE(pred_values, y_test))
hyper_values.append([estimators, depth, avg_acc_score])
avg_acc_score = sum(acc_score)/k
print('MSE of each fold:  {}'.format(acc_score))
print('Avg MSE: {}'.format(avg_acc_score))
```
The final results of the random forest regression were:
MSE of each fold:  [10.466084021899821, 16.84115857614325, 18.860320166543016, 23.864277401941745, 15.73568647050125, 13.776225070335666, 12.351881045004149, 17.23327814061998, 20.60788458255633, 27.00394602732672]
Avg MSE: 17.67407415028719.

![image](https://user-images.githubusercontent.com/76021844/153276691-be508348-8481-4969-9569-9d461c49de02.png)

### Final Comparison
  Despite the size and complexity of the random forest regressor model, the locally weighted linear regression yieled a lower final value for mean squared error, indicating that Lowess is the better model. However, it is important to note that this result cannot be fully generalized beyond this dataset and the exploratory work conducted in this paper yields results informed by the dataset used for training. That being said, the final MSE value for lowess was 17.65 and the final MSE for random forest regression was 17.67. When looking at the final plots for the Lowess model versus Random Forest, the random forest seemed to overfit to the data a little more than Lowess. A weakness of random forests in general is that they are quite sensitive to the data they are trained on, so that is another important consideration when analyzing the results of our experimentation. In terms of choosing a regression algorithm, however, it is important to note that all the training data is required when predicting with Lowess since it is a lazy learner algorithm. 






