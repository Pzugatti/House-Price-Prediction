# House-Price-Prediction
The aim of this project is to identify the suitable model to make the prediction for the house price with given significant predictor variables and used a supervised learning technique.

The data-set is available on Kaggle : https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

## Data-set examination
The data-set is CSV format shown below:
![dataset](https://user-images.githubusercontent.com/43289100/46279769-ac3c4080-c59c-11e8-933a-69d53edd12a3.PNG)

The train data-set has 1460 samples, 80 features and 1 target variable.
The test data-set has 1459 samples and 80 features.
The target variable is sale price.

## Training data pre-processing
The heatmap showed the correlation of each variable with another. The darker colour means the relationship between any two variables are strongly correlated and lighter colour means they have almost no relationship.
![all_numericvariable_heatmap](https://user-images.githubusercontent.com/43289100/46280304-27522680-c59e-11e8-8fa7-5022aa24ec87.png)


#### The selected top 10 variables that highly correlated with 'SalePrice' from heatmap shown below:

![topnumericvalue_heatmap](https://user-images.githubusercontent.com/43289100/46281780-31762400-c5a2-11e8-967c-80b2f06f60a3.png)

By observing the above heatmap, the provided data is sufferring multicollinerity such as 'TotRmsAbvGrd' with 'GrLivArea', 'TotalBsmtSF' with '1stFlrSF', 'GarageCars' with 'GarageArea' and more, which independent variable has the strong relationship with another independent variable and the model performance will be affected.

#### The selected top 10 variables in 'pairplot':
![topnumericvalue_pairplot](https://user-images.githubusercontent.com/43289100/46282836-283a8680-c5a5-11e8-9398-4daae4a71d4e.png)

By looking the 'SalePrice' on y-axis and compare with 'GrLivArea',  'TotalBsmtSF' and '1stFlrSF' on x-axis , they seem do have outliers. since the 'TotalBsmtSF' and '1stFlrSF' are strongly correlated, only the 'TotalBsmtSF' has been taken for further analysis and 'GrLivArea' as well.

#### Analyse outlier
![sp_grla_scatter_outlier](https://user-images.githubusercontent.com/43289100/46283804-2625f700-c5a8-11e8-82d2-6dadfc90f141.png)
![sp_totalbsf_scatter_outlier](https://user-images.githubusercontent.com/43289100/46283806-2625f700-c5a8-11e8-9f35-e9c3bc1504a4.png)

From above two images, we could clearly see that several data points (red circle) are lies on abnormal distance from other values in a random samples, this could cause a problem,heteroscedasticity, which forming a cone-like shape pattern result in standard error bias. The cone-like shape pattern (green lines) shown below:

![sp_grla_scatter_outlier_coneline](https://user-images.githubusercontent.com/43289100/46284268-b9abf780-c5a9-11e8-8097-ad7d61f067a6.png)
![sp_totalbsf_scatter_outlier_coneline](https://user-images.githubusercontent.com/43289100/46284267-b9abf780-c5a9-11e8-953e-f08dab346c5f.png)

#### Remove weak features
Those weak features with the correlation coefficient less than 0.2 that almost no relationship with the 'SalePrice' has been removed.

![removed_not_important_numericvariable_heatmap](https://user-images.githubusercontent.com/43289100/46284546-cd0b9280-c5aa-11e8-85b9-04304cc16148.png)

#### Display and impute missing data
Removing all these features with missing data aren't a good practice, because some of these features might be important. However, there will be a tidious work to fill up the missing data that depending on domain knowledge and experience. In order to keep the process simple, all the features with missing data have been removed except 'Electrical', which only one missing data. The missing data will be fill in with most common value in 'Electrical'.
![missingdata](https://user-images.githubusercontent.com/43289100/46284954-45268800-c5ac-11e8-80f2-bdd92c2f5fee.PNG)



