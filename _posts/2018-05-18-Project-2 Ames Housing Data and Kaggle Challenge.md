---
layout: post
title: Project-2 Ames Housing Data and Kaggle Challenge
date: 2018-05-18
---


```python
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```

## Define problem
1. Get knowledge of the most important elements that people care about when buying a house in Ames, IA
2. Predict the price of house at sale using the Ames Iowa Housing dataset

## Gather Data
Load the Ames Iowa Housing dataset


```python
# load the data
df = pd.read_csv("train.csv")
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2051 entries, 0 to 2050
    Data columns (total 81 columns):
    Id                 2051 non-null int64
    PID                2051 non-null int64
    MS SubClass        2051 non-null int64
    MS Zoning          2051 non-null object
    Lot Frontage       1721 non-null float64
    Lot Area           2051 non-null int64
    Street             2051 non-null object
    Alley              140 non-null object
    Lot Shape          2051 non-null object
    Land Contour       2051 non-null object
    Utilities          2051 non-null object
    Lot Config         2051 non-null object
    Land Slope         2051 non-null object
    Neighborhood       2051 non-null object
    Condition 1        2051 non-null object
    Condition 2        2051 non-null object
    Bldg Type          2051 non-null object
    House Style        2051 non-null object
    Overall Qual       2051 non-null int64
    Overall Cond       2051 non-null int64
    Year Built         2051 non-null int64
    Year Remod/Add     2051 non-null int64
    Roof Style         2051 non-null object
    Roof Matl          2051 non-null object
    Exterior 1st       2051 non-null object
    Exterior 2nd       2051 non-null object
    Mas Vnr Type       2029 non-null object
    Mas Vnr Area       2029 non-null float64
    Exter Qual         2051 non-null object
    Exter Cond         2051 non-null object
    Foundation         2051 non-null object
    Bsmt Qual          1996 non-null object
    Bsmt Cond          1996 non-null object
    Bsmt Exposure      1993 non-null object
    BsmtFin Type 1     1996 non-null object
    BsmtFin SF 1       2050 non-null float64
    BsmtFin Type 2     1995 non-null object
    BsmtFin SF 2       2050 non-null float64
    Bsmt Unf SF        2050 non-null float64
    Total Bsmt SF      2050 non-null float64
    Heating            2051 non-null object
    Heating QC         2051 non-null object
    Central Air        2051 non-null object
    Electrical         2051 non-null object
    1st Flr SF         2051 non-null int64
    2nd Flr SF         2051 non-null int64
    Low Qual Fin SF    2051 non-null int64
    Gr Liv Area        2051 non-null int64
    Bsmt Full Bath     2049 non-null float64
    Bsmt Half Bath     2049 non-null float64
    Full Bath          2051 non-null int64
    Half Bath          2051 non-null int64
    Bedroom AbvGr      2051 non-null int64
    Kitchen AbvGr      2051 non-null int64
    Kitchen Qual       2051 non-null object
    TotRms AbvGrd      2051 non-null int64
    Functional         2051 non-null object
    Fireplaces         2051 non-null int64
    Fireplace Qu       1051 non-null object
    Garage Type        1938 non-null object
    Garage Yr Blt      1937 non-null float64
    Garage Finish      1937 non-null object
    Garage Cars        2050 non-null float64
    Garage Area        2050 non-null float64
    Garage Qual        1937 non-null object
    Garage Cond        1937 non-null object
    Paved Drive        2051 non-null object
    Wood Deck SF       2051 non-null int64
    Open Porch SF      2051 non-null int64
    Enclosed Porch     2051 non-null int64
    3Ssn Porch         2051 non-null int64
    Screen Porch       2051 non-null int64
    Pool Area          2051 non-null int64
    Pool QC            9 non-null object
    Fence              400 non-null object
    Misc Feature       65 non-null object
    Misc Val           2051 non-null int64
    Mo Sold            2051 non-null int64
    Yr Sold            2051 non-null int64
    Sale Type          2051 non-null object
    SalePrice          2051 non-null int64
    dtypes: float64(11), int64(28), object(42)
    memory usage: 1.3+ MB


## Explore & Clean Data
1. Check null values
2. Check data type for each columns
3. Split features and target
3. Get dummy variables for categorical features (for training / testing / predicting dataset together)
4. Training / Testing dataset splitting


```python
df.isnull().sum().sort_values(ascending=False)
```




    Pool QC            2042
    Misc Feature       1986
    Alley              1911
    Fence              1651
    Fireplace Qu       1000
    Lot Frontage        330
    Garage Finish       114
    Garage Cond         114
    Garage Qual         114
    Garage Yr Blt       114
    Garage Type         113
    Bsmt Exposure        58
    BsmtFin Type 2       56
    BsmtFin Type 1       55
    Bsmt Cond            55
    Bsmt Qual            55
    Mas Vnr Type         22
    Mas Vnr Area         22
    Bsmt Half Bath        2
    Bsmt Full Bath        2
    Garage Cars           1
    Garage Area           1
    Bsmt Unf SF           1
    BsmtFin SF 2          1
    Total Bsmt SF         1
    BsmtFin SF 1          1
    Overall Cond          0
    Exterior 2nd          0
    Exterior 1st          0
    Roof Matl             0
                       ... 
    Heating               0
    Exter Cond            0
    TotRms AbvGrd         0
    Yr Sold               0
    Mo Sold               0
    Misc Val              0
    Pool Area             0
    Screen Porch          0
    3Ssn Porch            0
    Enclosed Porch        0
    Open Porch SF         0
    Wood Deck SF          0
    Paved Drive           0
    Fireplaces            0
    Functional            0
    Kitchen Qual          0
    Foundation            0
    Kitchen AbvGr         0
    Bedroom AbvGr         0
    Half Bath             0
    Full Bath             0
    Gr Liv Area           0
    Low Qual Fin SF       0
    2nd Flr SF            0
    1st Flr SF            0
    Electrical            0
    Central Air           0
    Heating QC            0
    Sale Type             0
    Id                    0
    Length: 81, dtype: int64




```python
# Based on my investigation about the "null" values, the null values in numeric columns should be 0,
# The null values in categorical columns are the same as "None" or "doesn't have", which we can also use "0" to replace
df.fillna(value=0,inplace=True)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2051 entries, 0 to 2050
    Data columns (total 81 columns):
    Id                 2051 non-null int64
    PID                2051 non-null int64
    MS SubClass        2051 non-null int64
    MS Zoning          2051 non-null object
    Lot Frontage       2051 non-null float64
    Lot Area           2051 non-null int64
    Street             2051 non-null object
    Alley              2051 non-null object
    Lot Shape          2051 non-null object
    Land Contour       2051 non-null object
    Utilities          2051 non-null object
    Lot Config         2051 non-null object
    Land Slope         2051 non-null object
    Neighborhood       2051 non-null object
    Condition 1        2051 non-null object
    Condition 2        2051 non-null object
    Bldg Type          2051 non-null object
    House Style        2051 non-null object
    Overall Qual       2051 non-null int64
    Overall Cond       2051 non-null int64
    Year Built         2051 non-null int64
    Year Remod/Add     2051 non-null int64
    Roof Style         2051 non-null object
    Roof Matl          2051 non-null object
    Exterior 1st       2051 non-null object
    Exterior 2nd       2051 non-null object
    Mas Vnr Type       2051 non-null object
    Mas Vnr Area       2051 non-null float64
    Exter Qual         2051 non-null object
    Exter Cond         2051 non-null object
    Foundation         2051 non-null object
    Bsmt Qual          2051 non-null object
    Bsmt Cond          2051 non-null object
    Bsmt Exposure      2051 non-null object
    BsmtFin Type 1     2051 non-null object
    BsmtFin SF 1       2051 non-null float64
    BsmtFin Type 2     2051 non-null object
    BsmtFin SF 2       2051 non-null float64
    Bsmt Unf SF        2051 non-null float64
    Total Bsmt SF      2051 non-null float64
    Heating            2051 non-null object
    Heating QC         2051 non-null object
    Central Air        2051 non-null object
    Electrical         2051 non-null object
    1st Flr SF         2051 non-null int64
    2nd Flr SF         2051 non-null int64
    Low Qual Fin SF    2051 non-null int64
    Gr Liv Area        2051 non-null int64
    Bsmt Full Bath     2051 non-null float64
    Bsmt Half Bath     2051 non-null float64
    Full Bath          2051 non-null int64
    Half Bath          2051 non-null int64
    Bedroom AbvGr      2051 non-null int64
    Kitchen AbvGr      2051 non-null int64
    Kitchen Qual       2051 non-null object
    TotRms AbvGrd      2051 non-null int64
    Functional         2051 non-null object
    Fireplaces         2051 non-null int64
    Fireplace Qu       2051 non-null object
    Garage Type        2051 non-null object
    Garage Yr Blt      2051 non-null float64
    Garage Finish      2051 non-null object
    Garage Cars        2051 non-null float64
    Garage Area        2051 non-null float64
    Garage Qual        2051 non-null object
    Garage Cond        2051 non-null object
    Paved Drive        2051 non-null object
    Wood Deck SF       2051 non-null int64
    Open Porch SF      2051 non-null int64
    Enclosed Porch     2051 non-null int64
    3Ssn Porch         2051 non-null int64
    Screen Porch       2051 non-null int64
    Pool Area          2051 non-null int64
    Pool QC            2051 non-null object
    Fence              2051 non-null object
    Misc Feature       2051 non-null object
    Misc Val           2051 non-null int64
    Mo Sold            2051 non-null int64
    Yr Sold            2051 non-null int64
    Sale Type          2051 non-null object
    SalePrice          2051 non-null int64
    dtypes: float64(11), int64(28), object(42)
    memory usage: 1.3+ MB



```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Overall Qual</th>
      <th>Overall Cond</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Mas Vnr Area</th>
      <th>...</th>
      <th>Wood Deck SF</th>
      <th>Open Porch SF</th>
      <th>Enclosed Porch</th>
      <th>3Ssn Porch</th>
      <th>Screen Porch</th>
      <th>Pool Area</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2051.000000</td>
      <td>2.051000e+03</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>...</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1474.033642</td>
      <td>7.135900e+08</td>
      <td>57.008776</td>
      <td>57.944417</td>
      <td>10065.208191</td>
      <td>6.112140</td>
      <td>5.562165</td>
      <td>1971.708922</td>
      <td>1984.190151</td>
      <td>98.626524</td>
      <td>...</td>
      <td>93.833740</td>
      <td>47.556802</td>
      <td>22.571916</td>
      <td>2.591419</td>
      <td>16.511458</td>
      <td>2.397855</td>
      <td>51.574354</td>
      <td>6.219893</td>
      <td>2007.775719</td>
      <td>181469.701609</td>
    </tr>
    <tr>
      <th>std</th>
      <td>843.980841</td>
      <td>1.886918e+08</td>
      <td>42.824223</td>
      <td>33.137332</td>
      <td>6742.488909</td>
      <td>1.426271</td>
      <td>1.104497</td>
      <td>30.177889</td>
      <td>21.036250</td>
      <td>174.324690</td>
      <td>...</td>
      <td>128.549416</td>
      <td>66.747241</td>
      <td>59.845110</td>
      <td>25.229615</td>
      <td>57.374204</td>
      <td>37.782570</td>
      <td>573.393985</td>
      <td>2.744736</td>
      <td>1.312014</td>
      <td>79258.659352</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>5.263011e+08</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>12789.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>753.500000</td>
      <td>5.284581e+08</td>
      <td>20.000000</td>
      <td>43.500000</td>
      <td>7500.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1953.500000</td>
      <td>1964.500000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>2007.000000</td>
      <td>129825.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1486.000000</td>
      <td>5.354532e+08</td>
      <td>50.000000</td>
      <td>63.000000</td>
      <td>9430.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1974.000000</td>
      <td>1993.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>162500.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2198.000000</td>
      <td>9.071801e+08</td>
      <td>70.000000</td>
      <td>78.000000</td>
      <td>11513.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2001.000000</td>
      <td>2004.000000</td>
      <td>159.000000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>70.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2930.000000</td>
      <td>9.241520e+08</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>159000.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>...</td>
      <td>1424.000000</td>
      <td>547.000000</td>
      <td>432.000000</td>
      <td>508.000000</td>
      <td>490.000000</td>
      <td>800.000000</td>
      <td>17000.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>611657.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 39 columns</p>
</div>




```python
# rename df columns
df.columns = [x.lower().replace(' ','_') for x in df.columns] 
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>pid</th>
      <th>ms_subclass</th>
      <th>ms_zoning</th>
      <th>lot_frontage</th>
      <th>lot_area</th>
      <th>street</th>
      <th>alley</th>
      <th>lot_shape</th>
      <th>land_contour</th>
      <th>...</th>
      <th>screen_porch</th>
      <th>pool_area</th>
      <th>pool_qc</th>
      <th>fence</th>
      <th>misc_feature</th>
      <th>misc_val</th>
      <th>mo_sold</th>
      <th>yr_sold</th>
      <th>sale_type</th>
      <th>saleprice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>109</td>
      <td>533352170</td>
      <td>60</td>
      <td>RL</td>
      <td>0.0</td>
      <td>13517</td>
      <td>Pave</td>
      <td>0</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>130500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>544</td>
      <td>531379050</td>
      <td>60</td>
      <td>RL</td>
      <td>43.0</td>
      <td>11492</td>
      <td>Pave</td>
      <td>0</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2009</td>
      <td>WD</td>
      <td>220000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>153</td>
      <td>535304180</td>
      <td>20</td>
      <td>RL</td>
      <td>68.0</td>
      <td>7922</td>
      <td>Pave</td>
      <td>0</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>WD</td>
      <td>109000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>318</td>
      <td>916386060</td>
      <td>60</td>
      <td>RL</td>
      <td>73.0</td>
      <td>9802</td>
      <td>Pave</td>
      <td>0</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>174000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>255</td>
      <td>906425045</td>
      <td>50</td>
      <td>RL</td>
      <td>82.0</td>
      <td>14235</td>
      <td>Pave</td>
      <td>0</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>138500</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
# slice X as features and y as target
target = 'saleprice'
features = [x for x in df.columns if x != target]
X = df[features]
y = df[target]
X.shape
```




    (2051, 80)




```python
X.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>pid</th>
      <th>ms_subclass</th>
      <th>ms_zoning</th>
      <th>lot_frontage</th>
      <th>lot_area</th>
      <th>street</th>
      <th>alley</th>
      <th>lot_shape</th>
      <th>land_contour</th>
      <th>...</th>
      <th>3ssn_porch</th>
      <th>screen_porch</th>
      <th>pool_area</th>
      <th>pool_qc</th>
      <th>fence</th>
      <th>misc_feature</th>
      <th>misc_val</th>
      <th>mo_sold</th>
      <th>yr_sold</th>
      <th>sale_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2046</th>
      <td>1587</td>
      <td>921126030</td>
      <td>20</td>
      <td>RL</td>
      <td>79.0</td>
      <td>11449</td>
      <td>Pave</td>
      <td>0</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>WD</td>
    </tr>
    <tr>
      <th>2047</th>
      <td>785</td>
      <td>905377130</td>
      <td>30</td>
      <td>RL</td>
      <td>0.0</td>
      <td>12342</td>
      <td>Pave</td>
      <td>0</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2009</td>
      <td>WD</td>
    </tr>
    <tr>
      <th>2048</th>
      <td>916</td>
      <td>909253010</td>
      <td>50</td>
      <td>RL</td>
      <td>57.0</td>
      <td>7558</td>
      <td>Pave</td>
      <td>0</td>
      <td>Reg</td>
      <td>Bnk</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2009</td>
      <td>WD</td>
    </tr>
    <tr>
      <th>2049</th>
      <td>639</td>
      <td>535179160</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>10400</td>
      <td>Pave</td>
      <td>0</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>2009</td>
      <td>WD</td>
    </tr>
    <tr>
      <th>2050</th>
      <td>10</td>
      <td>527162130</td>
      <td>60</td>
      <td>RL</td>
      <td>60.0</td>
      <td>7500</td>
      <td>Pave</td>
      <td>0</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>




```python
# load the predicting dataset in for dummy creation
predict = pd.read_csv("test.csv")

# clean up predicting dataset a little bit
predict.columns = [x.lower().replace(' ','_') for x in predict.columns]
predict.fillna(value=0,inplace=True)
predict.shape
```




    (879, 80)




```python
# concat training and predicting data together and get dummy variables
total = pd.concat([X,predict],axis=0)
total = pd.get_dummies(total,drop_first=True)
total.shape
```




    (2930, 274)




```python
# split training and predicting dataset to different dataframe
X = total.iloc[:len(X),:]
predict = total.iloc[len(X):,:]
X.shape, predict.shape
```




    ((2051, 274), (879, 274))




```python
X.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>pid</th>
      <th>ms_subclass</th>
      <th>lot_frontage</th>
      <th>lot_area</th>
      <th>overall_qual</th>
      <th>overall_cond</th>
      <th>year_built</th>
      <th>year_remod/add</th>
      <th>mas_vnr_area</th>
      <th>...</th>
      <th>misc_feature_TenC</th>
      <th>sale_type_CWD</th>
      <th>sale_type_Con</th>
      <th>sale_type_ConLD</th>
      <th>sale_type_ConLI</th>
      <th>sale_type_ConLw</th>
      <th>sale_type_New</th>
      <th>sale_type_Oth</th>
      <th>sale_type_VWD</th>
      <th>sale_type_WD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2046</th>
      <td>1587</td>
      <td>921126030</td>
      <td>20</td>
      <td>79.0</td>
      <td>11449</td>
      <td>8</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2047</th>
      <td>785</td>
      <td>905377130</td>
      <td>30</td>
      <td>0.0</td>
      <td>12342</td>
      <td>4</td>
      <td>5</td>
      <td>1940</td>
      <td>1950</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2048</th>
      <td>916</td>
      <td>909253010</td>
      <td>50</td>
      <td>57.0</td>
      <td>7558</td>
      <td>6</td>
      <td>6</td>
      <td>1928</td>
      <td>1950</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2049</th>
      <td>639</td>
      <td>535179160</td>
      <td>20</td>
      <td>80.0</td>
      <td>10400</td>
      <td>4</td>
      <td>5</td>
      <td>1956</td>
      <td>1956</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2050</th>
      <td>10</td>
      <td>527162130</td>
      <td>60</td>
      <td>60.0</td>
      <td>7500</td>
      <td>7</td>
      <td>5</td>
      <td>1999</td>
      <td>1999</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 274 columns</p>
</div>




```python
# training / testing data set spliting
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,shuffle=True)
```

## Feature Engineering on Training Dataset
1. Manually drop "id, pid" columns 
2. Get polynomial features
3. Standardize the features
4. Decide to use Lasso regression model to help feature selections by eliminating the number of features
2. After get the"important features", save the feature names for further modeling


```python
# drop ids which won't be a good predictor
X_train.drop(['id','pid'],axis=1,inplace=True)
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/ipykernel/__main__.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from ipykernel import kernelapp as app



```python
# creat PolynomialFeatures for training data
poly = PolynomialFeatures(include_bias=False,degree=2)
X_train_poly = poly.fit_transform(X_train)
X_train_poly.shape
```




    (1538, 37400)




```python
X_train_poly_df=pd.DataFrame(X_train_poly,columns=poly.get_feature_names(X_train.columns))
X_train_poly_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ms_subclass</th>
      <th>lot_frontage</th>
      <th>lot_area</th>
      <th>overall_qual</th>
      <th>overall_cond</th>
      <th>year_built</th>
      <th>year_remod/add</th>
      <th>mas_vnr_area</th>
      <th>bsmtfin_sf_1</th>
      <th>bsmtfin_sf_2</th>
      <th>...</th>
      <th>sale_type_New^2</th>
      <th>sale_type_New sale_type_Oth</th>
      <th>sale_type_New sale_type_VWD</th>
      <th>sale_type_New sale_type_WD</th>
      <th>sale_type_Oth^2</th>
      <th>sale_type_Oth sale_type_VWD</th>
      <th>sale_type_Oth sale_type_WD</th>
      <th>sale_type_VWD^2</th>
      <th>sale_type_VWD sale_type_WD</th>
      <th>sale_type_WD ^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20.0</td>
      <td>85.0</td>
      <td>10667.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>1971.0</td>
      <td>1971.0</td>
      <td>302.0</td>
      <td>838.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>70.0</td>
      <td>107.0</td>
      <td>12888.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>1937.0</td>
      <td>1980.0</td>
      <td>0.0</td>
      <td>288.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20.0</td>
      <td>60.0</td>
      <td>7200.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>1950.0</td>
      <td>2002.0</td>
      <td>0.0</td>
      <td>398.0</td>
      <td>149.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.0</td>
      <td>80.0</td>
      <td>14000.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>1996.0</td>
      <td>1997.0</td>
      <td>0.0</td>
      <td>1201.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60.0</td>
      <td>0.0</td>
      <td>11929.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>1995.0</td>
      <td>1995.0</td>
      <td>466.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37400 columns</p>
</div>




```python
# standarize the features
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train_poly)
X_train_scaled.shape
```




    (1538, 37400)




```python
X_train_scaled_df=pd.DataFrame(X_train_scaled,columns=poly.get_feature_names(X_train.columns))
X_train_scaled_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ms_subclass</th>
      <th>lot_frontage</th>
      <th>lot_area</th>
      <th>overall_qual</th>
      <th>overall_cond</th>
      <th>year_built</th>
      <th>year_remod/add</th>
      <th>mas_vnr_area</th>
      <th>bsmtfin_sf_1</th>
      <th>bsmtfin_sf_2</th>
      <th>...</th>
      <th>sale_type_New^2</th>
      <th>sale_type_New sale_type_Oth</th>
      <th>sale_type_New sale_type_VWD</th>
      <th>sale_type_New sale_type_WD</th>
      <th>sale_type_Oth^2</th>
      <th>sale_type_Oth sale_type_VWD</th>
      <th>sale_type_Oth sale_type_WD</th>
      <th>sale_type_VWD^2</th>
      <th>sale_type_VWD sale_type_WD</th>
      <th>sale_type_WD ^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.856027</td>
      <td>0.798505</td>
      <td>0.131102</td>
      <td>-0.076947</td>
      <td>0.400164</td>
      <td>-0.028539</td>
      <td>-0.629478</td>
      <td>1.159913</td>
      <td>0.840886</td>
      <td>-0.283011</td>
      <td>...</td>
      <td>-0.285614</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.044209</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.388841</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.301295</td>
      <td>1.448310</td>
      <td>0.540450</td>
      <td>0.623315</td>
      <td>2.218337</td>
      <td>-1.155710</td>
      <td>-0.199259</td>
      <td>-0.558774</td>
      <td>-0.331752</td>
      <td>-0.283011</td>
      <td>...</td>
      <td>-0.285614</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.044209</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.388841</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.856027</td>
      <td>0.060091</td>
      <td>-0.507892</td>
      <td>-0.777209</td>
      <td>2.218337</td>
      <td>-0.724733</td>
      <td>0.852389</td>
      <td>-0.558774</td>
      <td>-0.097224</td>
      <td>0.593720</td>
      <td>...</td>
      <td>-0.285614</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.044209</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.388841</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.069831</td>
      <td>0.650822</td>
      <td>0.745400</td>
      <td>0.623315</td>
      <td>-0.508923</td>
      <td>0.800262</td>
      <td>0.613378</td>
      <td>-0.558774</td>
      <td>1.614827</td>
      <td>-0.283011</td>
      <td>...</td>
      <td>-0.285614</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.044209</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.388841</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.069831</td>
      <td>-1.712103</td>
      <td>0.363699</td>
      <td>1.323577</td>
      <td>1.309251</td>
      <td>0.767110</td>
      <td>0.517774</td>
      <td>2.093240</td>
      <td>-0.945787</td>
      <td>-0.283011</td>
      <td>...</td>
      <td>-0.285614</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.044209</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.388841</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37400 columns</p>
</div>




```python
# Because there are too many predictors, trying to build a Lasso regression model and select the right features
# Find an optimal value for lasso regression alpha using LassoCV
lasso_model = LassoCV(n_alphas=100)
lasso_model = lasso_model.fit(X_train_scaled, y_train)
lasso_optimal_alpha = lasso_model.alpha_
lasso_optimal_alpha
```




    812.0487670259105




```python
# build a lasso model using the optimal alpha, check the cross-validation score
lasso_model = Lasso(alpha=lasso_optimal_alpha)
cross_val_score(lasso_model, X_train_scaled, y_train).mean()
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)





    0.7983608634379319




```python
# fit the model
lasso_model = lasso_model.fit(X_train_scaled, y_train)
```


```python
# check the lasso coefficients - 368 left
lasso_coef = pd.Series(data=lasso_model.coef_, index=poly.get_feature_names(X_train.columns))
lasso_coef = pd.DataFrame(data=lasso_coef)
lasso_coef.rename(columns={0:'lasso_coef'},inplace=True)
lasso_coef['lasso_coef_abs'] = lasso_coef['lasso_coef'].abs()
lasso_coef.sort_values(by='lasso_coef_abs',ascending=False)
features_picked = lasso_coef[lasso_coef.lasso_coef!=0].index
features_picked
```




    Index(['ms_subclass^2', 'ms_subclass ms_zoning_RM',
           'ms_subclass lot_shape_Reg', 'ms_subclass bsmt_exposure_No',
           'lot_frontage full_bath', 'lot_frontage half_bath',
           'lot_frontage garage_cars', 'lot_frontage exterior_2nd_CmentBd',
           'lot_frontage bsmtfin_type_2_GLQ', 'lot_frontage sale_type_Oth',
           ...
           'bsmtfin_type_1_Unf functional_Maj2',
           'bsmtfin_type_1_Unf sale_type_Oth',
           'bsmtfin_type_2_ALQ fireplace_qu_TA',
           'bsmtfin_type_2_ALQ garage_finish_Fin',
           'bsmtfin_type_2_Unf functional_Maj2',
           'bsmtfin_type_2_Unf garage_type_2Types',
           'kitchen_qual_Gd misc_feature_Othr', 'fireplace_qu_Gd pool_qc_Gd',
           'garage_type_Detchd sale_type_Oth', 'pool_qc_Gd sale_type_New'],
          dtype='object', length=368)




```python
# slice X_train_scaled data to the picked predictors
X_train_scaled_picked = X_train_scaled_df[features_picked]
X_train_scaled_picked.shape
```




    (1538, 368)



## Model with data & Evaluate
A. Modeling  
1. Use Cross-validation to try Linear / Lasso / Ridge / Elastic Net regression model and see which one perform better on the training dataset
2. Compared the cross-validation results, Ridge is the best, but we still want to test them on the test dataset to make sure they are not overfitting  

B. Evaluation  
1. Clean up testing dataset - Same process to training dataset
2. Feature engineering on testing dataset - Same process to training dataset
3. Fit the models using training data and use the testing dataset to score each models (Linear / Lasso / Ridge / Elastic Net regression) to see which one has the highest score 
4. Lasso got the highest score on testing dataset (0.93 r2 score), which is pretty good. So I decided to use this model to do prediction

C. Optimization  
1. After decision (Lasso regression model), for optimization purpose, I'm building the real model using the entire training + testing dataset I have to make a better model (lasso_model_2)


### A. Modeling


```python
# Try LinearRegression for the picked features, it's so bad
lr = LinearRegression()
lr_model = lr.fit(X_train_scaled_picked,y_train)
cross_val_score(lr_model, X_train_scaled_picked, y_train).mean()
```




    -3.1510322321266766e+22




```python
# Try Lasso Regression
lasso_model_1 = LassoCV(n_alphas=200)
lasso_model_1 = lasso_model_1.fit(X_train_scaled_picked, y_train)
lasso_optimal_alpha_1 = lasso_model_1.alpha_
lasso_optimal_alpha_1
```




    65.86777910394683




```python
lasso_model_1 = Lasso(alpha=lasso_optimal_alpha,max_iter=10000)
cross_val_score(lasso_model_1, X_train_scaled_picked, y_train).mean()
```




    0.8414126748286502




```python
# Try Ridge Regression
r_alphas = np.logspace(0, 5, 200) 
ridge_model = RidgeCV(alphas=r_alphas, store_cv_values=True)
ridge_model = ridge_model.fit(X_train_scaled_picked, y_train)
ridge_optimal_alpha = ridge_model.alpha_
ridge_optimal_alpha
```




    1.0




```python
ridge_opt = Ridge(alpha=ridge_optimal_alpha)
cross_val_score(ridge_opt, X_train_scaled_picked, y_train).mean()
```




    0.9155796608593949




```python
# Try Elastic Net Regression, enet_ratio = 1 is the best (tried few times)
enet_ratio = 1
enet_model = ElasticNetCV(n_alphas=100, l1_ratio=enet_ratio)
enet_model = enet_model.fit(X_train_scaled_picked, y_train)
enet_optimal_alpha = enet_model.alpha_
enet_optimal_alpha
```




    65.86777910394683




```python
enet_model = ElasticNet(alpha=enet_optimal_alpha,l1_ratio=enet_ratio)
cross_val_score(enet_model, X_train_scaled_picked, y_train).mean()
```




    0.8926195740654571



### B. Evaluation


```python
# drop testing data set id and pid which won't be a good predictors
X_test.drop(['id','pid'],axis=1,inplace=True)
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/ipykernel/__main__.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from ipykernel import kernelapp as app



```python
# creat PolynomialFeatures for testing data
X_test_poly = PolynomialFeatures(include_bias=False,degree=2)
X_test_poly = poly.fit_transform(X_test)
X_test_poly.shape
```




    (513, 37400)




```python
# standarize the features
ss = StandardScaler()
X_test_scaled = ss.fit_transform(X_test_poly)
X_test_scaled.shape
```




    (513, 37400)




```python
# slice the X_test_scaled data to only the selected features I picked using Lasso feature selection
X_test_scaled_df=pd.DataFrame(X_test_scaled,columns=poly.get_feature_names(X_test.columns))
X_test_scaled_picked = X_test_scaled_df[features_picked]
```


```python
# Fit and Score Ridge Regression
ridge_model = ridge_model.fit(X_train_scaled_picked, y_train)
ridge_model.score(X_test_scaled_picked,y_test)
```




    0.9163585368742567




```python
# Fit and Score Elastic Net Regression
enet_model = enet_model.fit(X_train_scaled_picked, y_train)
enet_model.score(X_test_scaled_picked,y_test)
```




    0.9190279024672294




```python
# Fit and Score lasso model on testing data
lasso_model_1 = lasso_model_1.fit(X_train_scaled_picked, y_train)
lasso_model_1.score(X_test_scaled_picked,y_test)
```




    0.9298597791294881



### C. Optimization


```python
X_scaled_picked = pd.concat([X_train_scaled_picked,X_test_scaled_picked])
X_scaled_picked.shape
```




    (2051, 368)




```python
y_total = pd.concat([y_train,y_test])
y_total.shape
```




    (2051,)




```python
# Fit entire dataset and Score lasso test score
X_scaled_picked = pd.concat([X_train_scaled_picked,X_test_scaled_picked])
y_total = pd.concat([y_train,y_test])
lasso_model_2 = lasso_model_1.fit(X_scaled_picked, y_total)
lasso_model_2.score(X_test_scaled_picked,y_test)
```




    0.9515593750856383



## Answer problem

Questions:  
1. Get knowledge of the most important elements that people care about when buying a house in Ames, IA
2. Predict the price of house at sale using the Ames Iowa Housing dataset

Way to Answer:  
1. Check the Coefficients of the model
2. Use the test dataset to predict the sales price of Ames

Answer:  
![png](/images/Final_Explore_files/Final_Answer.png)


```python
# check the lasso coefficients
lasso_model_2 = lasso_model_1.fit(X_scaled_picked, y_total)
lasso_coef_2 = pd.Series(data=lasso_model_2.coef_, index=X_train_scaled_picked.columns)
lasso_coef_2 = pd.DataFrame(data=lasso_coef_2)
lasso_coef_2.rename(columns={0:'lasso_coef'},inplace=True)
lasso_coef_2['lasso_coef_abs'] = lasso_coef_2['lasso_coef'].abs()
lasso_coef_2.sort_values(by='lasso_coef_abs',ascending=False)
features_picked_2 = lasso_coef_2[lasso_coef_2.lasso_coef!=0].index
```


```python
lasso_coef_2.sort_values(by='lasso_coef_abs',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lasso_coef</th>
      <th>lasso_coef_abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>overall_qual gr_liv_area</th>
      <td>18645.918614</td>
      <td>18645.918614</td>
    </tr>
    <tr>
      <th>half_bath pool_qc_Gd</th>
      <td>-10982.224945</td>
      <td>10982.224945</td>
    </tr>
    <tr>
      <th>overall_cond gr_liv_area</th>
      <td>9019.628421</td>
      <td>9019.628421</td>
    </tr>
    <tr>
      <th>overall_qual garage_area</th>
      <td>5502.518096</td>
      <td>5502.518096</td>
    </tr>
    <tr>
      <th>year_built year_remod/add</th>
      <td>5431.422073</td>
      <td>5431.422073</td>
    </tr>
    <tr>
      <th>overall_qual total_bsmt_sf</th>
      <td>5170.781396</td>
      <td>5170.781396</td>
    </tr>
    <tr>
      <th>overall_qual misc_feature_Elev</th>
      <td>-3991.974233</td>
      <td>3991.974233</td>
    </tr>
    <tr>
      <th>overall_cond misc_feature_Elev</th>
      <td>-3913.613769</td>
      <td>3913.613769</td>
    </tr>
    <tr>
      <th>bsmtfin_sf_1 condition_1_Norm</th>
      <td>3829.189107</td>
      <td>3829.189107</td>
    </tr>
    <tr>
      <th>overall_qual fireplaces</th>
      <td>3724.080140</td>
      <td>3724.080140</td>
    </tr>
    <tr>
      <th>lot_area paved_drive_Y</th>
      <td>3425.209764</td>
      <td>3425.209764</td>
    </tr>
    <tr>
      <th>mas_vnr_area neighborhood_NridgHt</th>
      <td>3251.141657</td>
      <td>3251.141657</td>
    </tr>
    <tr>
      <th>gr_liv_area functional_Typ</th>
      <td>3067.565468</td>
      <td>3067.565468</td>
    </tr>
    <tr>
      <th>roof_style_Hip sale_type_New</th>
      <td>3066.384110</td>
      <td>3066.384110</td>
    </tr>
    <tr>
      <th>lot_area neighborhood_GrnHill</th>
      <td>2835.826324</td>
      <td>2835.826324</td>
    </tr>
    <tr>
      <th>year_built^2</th>
      <td>2637.945291</td>
      <td>2637.945291</td>
    </tr>
    <tr>
      <th>total_bsmt_sf foundation_PConc</th>
      <td>2405.257003</td>
      <td>2405.257003</td>
    </tr>
    <tr>
      <th>overall_qual 1st_flr_sf</th>
      <td>2177.213850</td>
      <td>2177.213850</td>
    </tr>
    <tr>
      <th>neighborhood_StoneBr fireplace_qu_Gd</th>
      <td>2066.659528</td>
      <td>2066.659528</td>
    </tr>
    <tr>
      <th>overall_qual garage_cars</th>
      <td>2065.134717</td>
      <td>2065.134717</td>
    </tr>
    <tr>
      <th>screen_porch bsmtfin_type_2_GLQ</th>
      <td>1960.133683</td>
      <td>1960.133683</td>
    </tr>
    <tr>
      <th>gr_liv_area ms_zoning_RM</th>
      <td>-1959.987916</td>
      <td>1959.987916</td>
    </tr>
    <tr>
      <th>mas_vnr_type_Stone bsmt_qual_Ex</th>
      <td>1888.975568</td>
      <td>1888.975568</td>
    </tr>
    <tr>
      <th>overall_qual bsmt_qual_Ex</th>
      <td>1843.163882</td>
      <td>1843.163882</td>
    </tr>
    <tr>
      <th>neighborhood_NridgHt bsmt_qual_Ex</th>
      <td>1802.671579</td>
      <td>1802.671579</td>
    </tr>
    <tr>
      <th>lot_area neighborhood_StoneBr</th>
      <td>1792.400928</td>
      <td>1792.400928</td>
    </tr>
    <tr>
      <th>gr_liv_area exter_qual_TA</th>
      <td>-1785.893957</td>
      <td>1785.893957</td>
    </tr>
    <tr>
      <th>lot_area overall_cond</th>
      <td>1770.573451</td>
      <td>1770.573451</td>
    </tr>
    <tr>
      <th>bsmtfin_sf_1 exterior_2nd_VinylSd</th>
      <td>1718.189672</td>
      <td>1718.189672</td>
    </tr>
    <tr>
      <th>bsmtfin_sf_1 bsmtfin_type_1_GLQ</th>
      <td>1689.427598</td>
      <td>1689.427598</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>bsmtfin_sf_2 mas_vnr_type_BrkCmn</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>bsmtfin_sf_2 exterior_2nd_Stucco</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>bsmtfin_sf_2 exterior_1st_Stucco</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_Timber fireplace_qu_Gd</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>condition_1_Norm exterior_1st_BrkFace</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>condition_1_PosA exterior_1st_MetalSd</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>condition_1_PosA exterior_2nd_MetalSd</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>bsmtfin_sf_1 bsmtfin_type_2_GLQ</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_NoRidge exterior_1st_HdBoard</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_NAmes fireplace_qu_TA</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_NAmes mas_vnr_type_BrkCmn</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_Mitchel bsmtfin_type_1_LwQ</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_ClearCr bsmt_qual_Ex</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2nd_flr_sf neighborhood_Crawfor</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_ClearCr garage_finish_RFn</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_CollgCr bldg_type_Duplex</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2nd_flr_sf ms_zoning_RM</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_Crawfor exterior_1st_VinylSd</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_Crawfor exterior_2nd_VinylSd</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1st_flr_sf bsmtfin_type_1_GLQ</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1st_flr_sf bsmt_qual_Ex</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_Crawfor garage_cond_Fa</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_Edwards exterior_1st_BrkComm</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_Edwards exterior_2nd_Brk Cmn</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_bsmt_sf functional_Typ</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_bsmt_sf land_contour_HLS</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_Edwards fireplace_qu_TA</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>neighborhood_Edwards pool_qc_Gd</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_bsmt_sf lot_shape_IR3</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>bsmt_unf_sf neighborhood_SawyerW</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>368 rows × 2 columns</p>
</div>




```python
# plot the predictions based on test dataset
y_test_hat = lasso_model_2.predict(X_test_scaled_picked)
sns.regplot(y_test_hat,y_test)
plt.xlabel('y_test_hat',fontsize=18)
plt.ylabel('y_test',fontsize=18)
```




    Text(0,0.5,'y_test')




![png](/images/Final_Explore_files/Final_Explore_51_1.png)



```python
# 225 out of 368 predictors are left, others are zero out
lasso_coef_2[lasso_coef_2['lasso_coef_abs']!=0].count()
```




    lasso_coef        225
    lasso_coef_abs    225
    dtype: int64




```python
lasso_coef_2.shape
```




    (368, 2)




```python
# creat PolynomialFeatures for predicting data
poly = PolynomialFeatures(include_bias=False,degree=2)
predict_poly = poly.fit_transform(predict)
predict_poly.shape
```




    (879, 37949)




```python
# standarize the features for predicting data
ss = StandardScaler()
predict_scaled = ss.fit_transform(predict_poly)
predict_scaled.shape
```




    (879, 37949)




```python
predict_scaled_df=pd.DataFrame(predict_scaled,columns=poly.get_feature_names(predict.columns))
predict_scaled_picked = predict_scaled_df[features_picked]
predict_scaled_picked.shape
```




    (879, 368)




```python
# predict y using predicting value
predict['saleprice'] = lasso_model_2.predict(predict_scaled_picked)
predict['saleprice'].shape
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/ipykernel/__main__.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from ipykernel import kernelapp as app





    (879,)




```python
# slice the id and saleprice columns from test dataset and save it to csv file
result = predict[['id','saleprice']]
result.rename(columns={"id": "Id", "saleprice": "SalePrice"},inplace=True)
result.set_index('Id',drop=True,inplace=True)
result.SalePrice = result.SalePrice.round(4)
result = result.sort_index(ascending=True)
result.to_csv('result.csv')
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/pandas/core/frame.py:3027: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      return super(DataFrame, self).rename(**kwargs)
    /anaconda3/envs/dsi/lib/python3.6/site-packages/pandas/core/generic.py:3643: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self[name] = value

