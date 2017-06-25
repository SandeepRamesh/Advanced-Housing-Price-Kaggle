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


#creating df for train and test and apply dummy variables
#dropped few extra because of shape differences in train and test
x_train=house_train.drop(['Id','SalePrice','Utilities','Condition2','RoofMatl'],1)
y_train=house_train['SalePrice']
x_test=house_test.drop(['Id'],1)

#convert categorical to numerical and avoid dummy variable trap
x_train=pd.get_dummies(x_train,drop_first=True)
x_test=pd.get_dummies(x_test,drop_first=True)
x_train=x_train.drop(['HouseStyle_2.5Fin'],1)
#Scaling the independent variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn import metrics
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
#Change it to required component after getting the most variance of the dataset
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_

sns.set()
pca = PCA().fit(x_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

from xgboost import XGBRegressor
regressor1 = XGBRegressor()
regressor1=regressor1.fit(x_train, y_train)
# Predicting the Test set results
y_pred1 = regressor1.predict(x_test)


results=pd.DataFrame({"Id":house_test['Id'],"SalePrice":y_pred1})
results.to_csv('Final_Output.csv',index=False)





