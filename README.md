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


### The selected top 10 variables that highly correlated with 'SalePrice' from heatmap shown below:

![topnumericvalue_heatmap](https://user-images.githubusercontent.com/43289100/46281780-31762400-c5a2-11e8-967c-80b2f06f60a3.png)

By observing the above heatmap, the provided data is sufferring multicollinerity such as 'TotRmsAbvGrd' with 'GrLivArea', 'TotalBsmtSF' with '1stFlrSF', 'GarageCars' with 'GarageArea' and more, which independent variable has the strong relationship with another independent variable and the model performance will be affected.

### The selected top 10 variables in 'pairplot':
![topnumericvalue_pairplot](https://user-images.githubusercontent.com/43289100/46282836-283a8680-c5a5-11e8-9398-4daae4a71d4e.png)

By looking the 'SalePrice' on y-axis and compare with 'GrLivArea',  'TotalBsmtSF' and '1stFlrSF' on x-axis , they seem do have outliers. since the 'TotalBsmtSF' and '1stFlrSF' are strongly correlated, only the 'TotalBsmtSF' has been taken for further analysis and 'GrLivArea' as well.

### Analyse outlier
![sp_totalbsf_scatter_outlier](https://user-images.githubusercontent.com/43289100/46283572-54ef9d80-c5a7-11e8-8630-9dbe7db930da.png)

![sp_grla_scatter_outlier](https://user-images.githubusercontent.com/43289100/46283566-50c38000-c5a7-11e8-9b8e-668096f6fa9b.png)

