# House-Price-Prediction
The aim of this project is to identify the suitable model to make the prediction for the house price with given significant predictor variables and used a supervised learning technique.

The data-set is available on Kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

## Data-set examination
#### The data-set is CSV format shown below:
![dataset](https://user-images.githubusercontent.com/43289100/46279769-ac3c4080-c59c-11e8-933a-69d53edd12a3.PNG)

* The train data-set has 1460 samples, 80 features and 1 target variable.
* The test data-set has 1459 samples and 80 features.
* The target variable is the sale price.

## Data-set pre-processing
The heat-map showed the correlation of each variable with another. The darker colour means the relationship between any two variables are strongly correlated and lighter colour means they have almost no relationship.
![all_numericvariable_heatmap](https://user-images.githubusercontent.com/43289100/46280304-27522680-c59e-11e8-8fa7-5022aa24ec87.png)


#### The selected top 10 variables that highly correlated to 'SalePrice' from heat-map shown below:

![topnumericvalue_heatmap](https://user-images.githubusercontent.com/43289100/46281780-31762400-c5a2-11e8-967c-80b2f06f60a3.png)

By observing the above heat-map, the provided data is sufferring multicollinearity such as 'TotRmsAbvGrd' with 'GrLivArea', 'TotalBsmtSF' with '1stFlrSF', 'GarageCars' with 'GarageArea' and more, which independent variable has the strong relationship with another independent variable and the model performance will be affected.

#### The selected top 10 variables in 'pairplot':
![topnumericvalue_pairplot](https://user-images.githubusercontent.com/43289100/46282836-283a8680-c5a5-11e8-9398-4daae4a71d4e.png)

By looking the 'SalePrice' on the y-axis and compare with 'GrLivArea',  'TotalBsmtSF' and '1stFlrSF' on the x-axis, they seem do have outliers. since the 'TotalBsmtSF' and '1stFlrSF' are strongly correlated, only the 'TotalBsmtSF' has been taken for further analysis and 'GrLivArea' as well.

#### Analyse outlier
![sp_grla_scatter_outlier](https://user-images.githubusercontent.com/43289100/46283804-2625f700-c5a8-11e8-82d2-6dadfc90f141.png)
![sp_totalbsf_scatter_outlier](https://user-images.githubusercontent.com/43289100/46283806-2625f700-c5a8-11e8-9f35-e9c3bc1504a4.png)

From above two images, we could clearly see that several data points (red circle) are lies on abnormal distance from other values in a random samples, this could cause a problem, heteroscedasticity, which forming a cone-like shape pattern result in standard error bias. The cone-like shape pattern (green lines) shown below:

![sp_grla_scatter_outlier_coneline](https://user-images.githubusercontent.com/43289100/46284268-b9abf780-c5a9-11e8-8097-ad7d61f067a6.png)
![sp_totalbsf_scatter_outlier_coneline](https://user-images.githubusercontent.com/43289100/46284267-b9abf780-c5a9-11e8-953e-f08dab346c5f.png)

There are a lot of techniques to deal with outliers, but, above mentioned outliers have been removed to keep process simpler. 

#### Remove weak features
Those weak features with the correlation coefficient less than 0.2 that almost no relationship with the 'SalePrice' have been removed.

![removed_not_important_numericvariable_heatmap](https://user-images.githubusercontent.com/43289100/46284546-cd0b9280-c5aa-11e8-85b9-04304cc16148.png)

#### Display and impute missing data
Removing all those features with missing data aren't a good practice, because some of those features might be important. However, there will be a tedious work to fill up the missing data that depending on domain knowledge and experience. In order to keep the process simple, all the features with missing data have been removed except 'Electrical', which only one missing data. The missing data will be filled in with the most common value in 'Electrical'.

![missingdata](https://user-images.githubusercontent.com/43289100/46284954-45268800-c5ac-11e8-80f2-bdd92c2f5fee.PNG)

#### Solving normality to prevent heteroscedasticity
The normality graph for 'SalePrice', 'GrLivArea' and 'TotalBsmtSF'. (Click on the picture to zoom in)
![normality](https://user-images.githubusercontent.com/43289100/46285814-62a92100-c5af-11e8-86cf-9421eb04f908.png)

This process is to transform the data into a normal distribution shape used the log transformation. The probability plot applied, which if the data points lie on the diagonal line, it means the particular feature more likely to be a normal distribution. Also, the skewness and kurtosis are indicating whether the feature is a normal, left skew, right skew distribution, heavier tails or light tails. 

According to the rule of thumb:
 - Reference: https://www.spcforexcel.com/knowledge/basic-statistics/are-skewness-and-kurtosis-useful-statistics
 - **Skewness**
      * If the skewness is between -0.5 and 0.5, the data are fairly symmetrical
      * If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed
      * If the skewness is less than -1 or greater than 1, the data are highly skewed
 - **Kurtosis**
      * If the kurtosis is close to 0, then a normal distribution is often assumed. These are called mesokurtic distributions.  
      * If the kurtosis is less than zero, then the distribution is light tails and is called a platykurtic distribution.  
      * If the kurtosis is greater than zero, then the distribution has heavier tails and is called a leptokurtic distribution.


#### The data has become less likely cone-like shape pattern:
![sp_grla_scatter_no_outlier](https://user-images.githubusercontent.com/43289100/46288416-25955c80-c5b8-11e8-9698-75faa8c9ddd1.png)
![sp_totalbsf_scatter_no_outlier](https://user-images.githubusercontent.com/43289100/46288417-262df300-c5b8-11e8-9959-1229d35abcff.png)


#### The remaining work
* The ordinal variables have been label encoding by converting a string into an ordered number.
* All year type variable's value converted into year interval. For instance: original = 1995 -> year interval = 23 = 2018 - 1995
* Lastly, all the data transformed into dummy values.
* The testing data-set has go through the similar process as training data-set does (e.g data transformation, remove weak features and etc.)


## Model training
The XGBoost model will be used in this project. XGBoost stands for e**X**treme **G**radient **B**oosting. XGBoost is fast and dominates structured or tabular datasets on classification and regression predictive modeling problems.
- The good explanation is here: https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/

The XGBoost's hyper-parameters have been randomly selected by using 'RandomizedSearchCV' library in order to get the best hyper-parameters with lesser execution time. After the training data fit into the XGBoost model, the result generated shown below:
- Note: 'Test' showed in the result is actually a validation data and 'r2' is R-squared.

![model_result](https://user-images.githubusercontent.com/43289100/46290370-38129480-c5be-11e8-8095-d2414cf37999.PNG)

The result shows that the 'Test r2' is slightly lower than 'Train r2', which means the model is little bit over-fitting.


#### Standardized residual shape pattern
![shape_of_standardized_residual](https://user-images.githubusercontent.com/43289100/46291411-f6cfb400-c5c0-11e8-8c0c-aff3219ab651.png)

#### QQ plot
visual checking whether the standardized residual is a normal distribution. The standardized residual seems like not close to normal distribution.

![qqplot](https://user-images.githubusercontent.com/43289100/46291482-33031480-c5c1-11e8-8b48-af81d6884ec0.png)

#### Residual plot
- reference: http://docs.statwing.com/interpreting-residual-plots-to-improve-your-regression/#y-unbalanced-header
![residualplot](https://user-images.githubusercontent.com/43289100/46291468-254d8f00-c5c1-11e8-82e3-25ad937de3ff.png)

The data points are not so evenly distributed vertically, the model has room for improvement.


## Predict unseen data-set (testing data-set)
#### The RMSLE (Root Mean Squared Logarithmic Error) that I obtained:
![kaggle_result](https://user-images.githubusercontent.com/43289100/46294110-a3ad2f80-c5c7-11e8-9ded-d1fdfea1cbef.PNG)

#### Other top 6 competitor results:
![top6_score](https://user-images.githubusercontent.com/43289100/46294256-f5ee5080-c5c7-11e8-8d7d-e78a84e2b787.PNG)


## Summary
After the result comparison, I still need to put more effort to improve the model. The result reflects that some of the valuable data might not yet to be discovered from the dataset. Probably need to review all the missing data, outliers, also, spend more time on data analysis and multicollinearity issue.

## Working enviroment
Google Colab
  - Python 3
  - xgboost 0.7.post4
  - sklearn 0.19.2
