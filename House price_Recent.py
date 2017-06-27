#Author: Sandeep Ramesh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from scipy import stats

house_train = pd.read_csv('train.csv')
house_test=pd.read_csv('test.csv')

#print(house_train.columns)
house_train.info()
house_test.info()
house_train.describe()
house_train=house_train.drop_duplicates()
house_test=house_test.drop_duplicates()
sns.heatmap(house_train.corr(method='spearman'))
#Taking care of continuous features first
#dropping the continuous features
house_train=house_train.drop(['MSSubClass','OverallCond','BsmtFinSF2','BsmtUnfSF',
                              'LowQualFinSF','BsmtFullBath','BsmtHalfBath','BedroomAbvGr',
                              'KitchenAbvGr','EnclosedPorch','3SsnPorch','ScreenPorch',
                              'PoolArea','MiscVal','MoSold','YrSold','PoolQC'],1)

house_test=house_test.drop(['MSSubClass','OverallCond','BsmtFinSF2','BsmtUnfSF',
                              'LowQualFinSF','BsmtFullBath','BsmtHalfBath','BedroomAbvGr',
                              'KitchenAbvGr','EnclosedPorch','3SsnPorch','ScreenPorch',
                              'PoolArea','MiscVal','MoSold','YrSold','PoolQC'],1)

#corr=house_train.select_dtypes(include=['float64','int64']).iloc[:,1:].corr()
#correlation between LotFrontage and LotArea
sns.jointplot(x='LotFrontage',y='LotArea',data=house_train)
#Got 2 outliers, now delete them
#index 1299, 935
house_train.sort_values(by='LotFrontage',ascending=False)[:2]
house_train=house_train.drop(house_train[house_train['Id']==1299].index)
house_train=house_train.drop(house_train[house_train['Id']==935].index)

#To fill in missing values
house_train['LotFrontage'].fillna(house_train['LotFrontage'].median(),inplace=True)
house_train.LotFrontage.isnull().sum()
#To check for normal distribution
sns.distplot(house_train.LotFrontage)

#remove outliers of LotArea
house_train.sort_values(by='LotArea',ascending=False)[:4]
house_train=house_train.drop(house_train[house_train['Id']==314].index)
house_train=house_train.drop(house_train[house_train['Id']==336].index)
house_train=house_train.drop(house_train[house_train['Id']==250].index)
house_train=house_train.drop(house_train[house_train['Id']==707].index)

#To check the null values sorted desc
#house_train.isnull().sum().sort_values(ascending=False).head(19)

#correlation between OverallQual and YearBuilt with SalePrice
sns.jointplot(x='OverallQual',y='SalePrice',data=house_train,kind='reg')
sns.jointplot(x='YearBuilt',y='SalePrice',data=house_train,kind='reg')
#2 outliers in SalePrice 
house_train=house_train.drop(house_train[house_train['Id']==692].index)
house_train=house_train.drop(house_train[house_train['Id']==1183].index)

#RemodAdd vs SalePrice since RemodAdd and YearBuilt are correlated
sns.jointplot(x='YearRemodAdd',y='SalePrice',data=house_train,kind='reg')
#6 outliers detected
house_train.sort_values(by='SalePrice',ascending=False)[:6]
#drop all the 6 outliers
house_train=house_train.drop(house_train[house_train['Id']==1170].index)
house_train=house_train.drop(house_train[house_train['Id']==899].index)
house_train=house_train.drop(house_train[house_train['Id']==804].index)
house_train=house_train.drop(house_train[house_train['Id']==1047].index)
house_train=house_train.drop(house_train[house_train['Id']==441].index)
house_train=house_train.drop(house_train[house_train['Id']==770].index)

#TotalBsmtSF and 1stFlrSF correlation
sns.jointplot(x='TotalBsmtSF',y='1stFlrSF',data=house_train,kind='reg')
#To remove outliers for TotalBsmtSF 
house_train=house_train.drop(house_train[house_train['Id']==333].index)
house_train=house_train.drop(house_train[house_train['Id']==524].index)

#GarageArea vs SalePrice
sns.jointplot(x='GarageArea',y='SalePrice',data=house_train,kind='reg')
#Remove outliers 
house_train=house_train.drop(house_train[house_train['Id']==582].index)
house_train=house_train.drop(house_train[house_train['Id']==1191].index)
house_train=house_train.drop(house_train[house_train['Id']==1062].index)

#Dropping the columns with mostly missing values
house_train=house_train.drop(['MiscFeature','Alley','Fence','FireplaceQu'],axis=1)

#now we will drop multicollinear features
house_train=house_train.drop(['1stFlrSF','GarageArea','GarageYrBlt'],axis=1)
#TotRmsAbvGrd and GrLivArea are multicollinear
house_train=house_train.drop(['TotRmsAbvGrd'],axis=1)
#2ndFlrSF and HalfBath are least correlated with SalePrice now, so we will drop those too
house_train=house_train.drop(['HalfBath','2ndFlrSF'],axis=1)

#Filling up missing values of continuous on test set
house_test['LotFrontage'].fillna(house_test['LotFrontage'].median(),inplace=True)
house_test.LotFrontage.isnull().sum()

house_train['MasVnrArea'].fillna(house_train['MasVnrArea'].median(),inplace=True)

#dropping in test set similar to training set
house_test=house_test.drop(['MiscFeature','Alley','Fence','FireplaceQu'],axis=1)
#now we will drop multicollinear features
house_test=house_test.drop(['1stFlrSF','GarageArea','GarageYrBlt'],axis=1)
#TotRmsAbvGrd and GrLivArea are multicollinear
house_test=house_test.drop(['TotRmsAbvGrd'],axis=1)
#2ndFlrSF and HalfBath are least correlated with SalePrice now, so we will drop those too
house_test=house_test.drop(['HalfBath','2ndFlrSF'],axis=1)
house_test['MasVnrArea'].fillna(house_test['MasVnrArea'].median(),inplace=True)

#Take Care of Categorical Features
'''
house_train.isnull().sum().sort_values(ascending=False)
Missing values
GarageType       81
GarageCond       81
GarageQual       81
GarageFinish     81
BsmtExposure     38
BsmtQual         37
BsmtCond         37
BsmtFinType1     37
BsmtFinType2     37
MasVnrType        8
Electrical        1
'''

#filling missing values of object datatypes
#sns.violinplot(x="GarageType", y="SalePrice", data=house_train, inner=None)
#sns.swarmplot(x="GarageType", y="SalePrice", data=house_train, color="r", alpha=.5)
sns.countplot(x="GarageType", data=house_train, hue="GarageCond")
sns.countplot(x="GarageCond", data=house_train)
sns.countplot(x="GarageQual", data=house_train, hue="GarageFinish")
sns.countplot(x="GarageFinish", data=house_train, hue="GarageQual")
#through interpretation of countplots
house_train['GarageType'] = house_train['GarageType'].fillna("Attchd")
house_train['GarageCond'] = house_train['GarageCond'].fillna("TA")
house_train['GarageQual'] = house_train['GarageQual'].fillna("Unf")
house_train['GarageFinish'] = house_train['GarageFinish'].fillna("TA")
#dropping the missing value in Electrical
house_train = house_train.drop(house_train.loc[house_train['Electrical'].isnull()].index)

#To fill in missing values of Bsmt Features
sns.countplot(x="BsmtExposure", data=house_train, hue="BsmtQual")
sns.countplot(x="BsmtExposure", data=house_train, hue="BsmtCond")
sns.countplot(x="BsmtQual", data=house_train, hue="BsmtCond")
sns.countplot(x="BsmtFinType1", data=house_train, hue="BsmtQual")
sns.countplot(x="BsmtFinType2", data=house_train, hue="BsmtQual")
#fill in with corresponding values of count plots
house_train['BsmtExposure'] = house_train['BsmtExposure'].fillna("No")
house_train['BsmtQual'] = house_train['BsmtQual'].fillna("TA")
house_train['BsmtCond'] = house_train['BsmtCond'].fillna("TA")
house_train['BsmtFinType1'] = house_train['BsmtFinType1'].fillna("Unf")
house_train['BsmtFinType2'] = house_train['BsmtFinType2'].fillna("Unf")
#dropping the rows with missing values
house_train = house_train.drop(house_train.loc[house_train['MasVnrType'].isnull()].index)

#Lets do the same for test data
house_test['GarageType'] = house_test['GarageType'].fillna("Attchd")
house_test['GarageCond'] = house_test['GarageCond'].fillna("TA")
house_test['GarageQual'] = house_test['GarageQual'].fillna("Unf")
house_test['GarageFinish'] = house_test['GarageFinish'].fillna("TA")
house_test['BsmtExposure'] = house_test['BsmtExposure'].fillna("No")
house_test['BsmtQual'] = house_test['BsmtQual'].fillna("TA")
house_test['BsmtCond'] = house_test['BsmtCond'].fillna("TA")
house_test['BsmtFinType1'] = house_test['BsmtFinType1'].fillna("Unf")
house_test['BsmtFinType2'] = house_test['BsmtFinType2'].fillna("Unf")

#filling in missing values of rest of the features in test set
#continuous variables with median as usual
house_test['BsmtFinSF1'].fillna(house_test['BsmtFinSF1'].median(),inplace=True)
house_test['TotalBsmtSF'].fillna(house_test['TotalBsmtSF'].median(),inplace=True)
house_test['GarageCars'].fillna(house_test['GarageCars'].median(),inplace=True)


sns.countplot(x="MasVnrType", data=house_test, palette="Greens_d")
sns.countplot(x="MSZoning", data=house_test, palette="Greens_d")
sns.countplot(x="Utilities", data=house_test, palette="Greens_d")
sns.countplot(x="Functional", data=house_test, palette="Greens_d")
sns.countplot(x="Exterior2nd", data=house_test, palette="Greens_d")
plt.xticks(rotation=70)
sns.countplot(x="KitchenQual", data=house_test, palette="Greens_d")
sns.countplot(x="SaleType", data=house_test, palette="Greens_d")
sns.countplot(x="Exterior1st", data=house_test, palette="Greens_d")
plt.xticks(rotation=70)

house_test['MasVnrType'] = house_test['MasVnrType'].fillna("None")
house_test['MSZoning'] = house_test['MSZoning'].fillna("RL")
house_test['Utilities'] = house_test['Utilities'].fillna("AllPub")
house_test['Functional'] = house_test['Functional'].fillna("Typ")
house_test['Exterior2nd'] = house_test['Exterior2nd'].fillna("VinylSd")
house_test['KitchenQual'] = house_test['KitchenQual'].fillna("TA")
house_test['SaleType'] = house_test['SaleType'].fillna("WD")
house_test['Exterior1st'] = house_test['Exterior1st'].fillna("VinylSd")


#Use RFE method to extract the best features alone and drop the rest

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
rfecv = RFECV(estimator=regressor, step=1, cv=StratifiedKFold(3))
rfecv.fit(x_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
cols=rfecv.ranking_
#new dataframe with features and their rankings
newcols=pd.DataFrame(x_train.columns,cols)

#Dropping the features with relation to their rankings obtained above
x_train=house_train.drop(['Id','SaleType','GarageCond','GarageQual','Functional','Electrical',
'Heating','BsmtFinType2','Foundation','Condition1','Neighborhood','Street'],1)

x_test=house_test.drop(['Id','SaleType','GarageCond','GarageQual','Functional','Electrical',
'Heating','BsmtFinType2','Foundation','Condition1','Neighborhood','Street'],1)
#Taking a log of Saleprice to transform 
y_train=np.log(house_train['SalePrice'])
#Creating dummy variables for categorical features
x_train=pd.get_dummies(x_train,drop_first=True)
x_test=pd.get_dummies(x_test,drop_first=True)

#Dropping the extra features since train and test had mismatch 
x_train=x_train.drop(['HouseStyle_2.5Fin','Exterior2nd_Other','Exterior1st_Stone','Exterior1st_ImStucc',
                      'RoofMatl_Roll','RoofMatl_Metal','RoofMatl_Membran',
                      'Condition2_RRAe','Condition2_RRAn','Condition2_RRNn',
                      'Utilities_NoSeWa','SalePrice'],1)


#Creating a train and test dataset for the original training set so that appropriate ML can be applied to test set later
X=house_train.drop(['Id','SaleType','GarageCond','GarageQual','Functional','Electrical',
'Heating','BsmtFinType2','Foundation','Condition1','Neighborhood','Street'],1)
Y=np.log(house_train['SalePrice'])
X=pd.get_dummies(X,drop_first=True)
X=X.drop(['HouseStyle_2.5Fin'],1)
X=X.drop(['Exterior2nd_Other','Exterior1st_Stone','Exterior1st_ImStucc',
                      'RoofMatl_Roll','RoofMatl_Metal','RoofMatl_Membran',
                      'Condition2_RRAe','Condition2_RRAn','Condition2_RRNn',
                      'Utilities_NoSeWa','SalePrice'],1)

#Split of training set and check the results
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, Y, random_state=42, test_size=.33)

from sklearn.linear_model import ElasticNet
clf2 = ElasticNet(alpha=0.0004, l1_ratio=1.2)

clf2.fit(X_train, y_train)
print ("R^2 is: \n", clf2.score(X_test, y_test))
elas_preds = clf2.predict(X_test)
print ('RMSE is: \n', mean_squared_error(y_test, elas_preds))
'''
RMSE is: 
 0.0132014324009
 '''

from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,color='b') 
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()
'''
Result of Linear:
R^2 is: 
 0.899040551834
RMSE is: 
 0.0141681927217
'''


from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor=regressor.fit(X_train, y_train)
print ("R^2 is: \n", regressor.score(X_test, y_test))
# Predicting the Test set results
predictions1 = regressor.predict(X_test)
print ('RMSE is: \n', mean_squared_error(y_test, predictions1))

'''
Result of XGB
R^2 is: 
 0.899040551834
RMSE is: 
 0.0139949162787
'''

from sklearn.model_selection import GridSearchCV
parameters = [{'max_depth': [10,20,30],
                'learning_rate': [0.05,0.07,1.0],
                'n_estimators':[50,100,200,150],
               # 'booster':['gbtree','gblinear','dart'],
                'gamma':[0.03,0.04,0.05],
                'reg_alpha':[0.2,0.4,0.6,0],
                'reg_lambda':[0.2,0.4,0.6,1]}
             ]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = None,
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

from sklearn.ensemble import RandomForestRegressor
regressor1 = RandomForestRegressor(n_estimators = 100, random_state = 42)
regressor1.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
# Predicting the Test set results
predictions2 = regressor1.predict(X_test)
print ('RMSE is: \n', mean_squared_error(y_test, predictions2))
'''
Result of Random Forest
R^2 is: 
 0.899040551834
RMSE is: 
 0.0159458246303
 '''
 
from sklearn.ensemble import ExtraTreesRegressor
regressor2=ExtraTreesRegressor(n_estimators=100,random_state=42)
regressor2.fit(X_train,y_train)
print ("R^2 is: \n", regressor2.score(X_test, y_test))
predictions3 = regressor2.predict(X_test)
print ('RMSE is: \n', mean_squared_error(y_test, predictions3))

'''
R^2 is: 
 0.874417216696
 RMSE is: 
 0.0176237203026
 '''
 
#ElasticNet has been chosen as it gave the best predictions. More algorithms will be tried later
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(x_train, y_train)
#Exponential taken to get the inverse of Log
predictions = np.exp(model.predict(x_test))

from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor=regressor.fit(x_train, y_train)
predictions1 = np.exp(regressor.predict(x_test))

from sklearn.linear_model import ElasticNet
clf2 = ElasticNet(alpha=0.0004, l1_ratio=1.2)
clf2.fit(x_train, y_train)
elas_preds = np.exp(clf2.predict(x_test))


#averaging the best 2 for final result
#final=(elas_preds+predictions1)/2

results=pd.DataFrame({"Id":house_test['Id'],"SalePrice":elas_preds})
results.to_csv('Final_Output.csv',index=False)
