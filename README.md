# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
DEVELOPED BY :NARRA RAMYA

REG NO :212223040128

```
 import pandas as pd
 from scipy import stats
 import numpy as np
```
```
 df=pd.read_csv("/content/bmi.csv")
 df.head()
```
![image](https://github.com/user-attachments/assets/caea494b-269e-4303-a994-a62cbdf15a9a)
```
 df_null_sum=df.isnull().sum()
 df_null_sum
```
![image](https://github.com/user-attachments/assets/9ee1d1f4-35b3-49a8-ad82-d31aa16eb0b5)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/92abe1ee-1f55-4bd2-a507-222fea9a938f)
```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
![image](https://github.com/user-attachments/assets/a948e606-6b02-496d-8dad-f731a27ee445)
```
 from sklearn.preprocessing import StandardScaler
 df1=pd.read_csv("/content/bmi.csv")
 df1.head()
```
![image](https://github.com/user-attachments/assets/5ca43d8d-70da-4f66-8b18-df21c00b503d)
```
 sc=StandardScaler()
 df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
 df1.head(10)
```
![image](https://github.com/user-attachments/assets/4e3ac267-a3a5-4631-a117-50e7793e5c1d)
```
 from sklearn.preprocessing import MinMaxScaler
 scaler=MinMaxScaler()
 df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
 df.head(10)
```
![image](https://github.com/user-attachments/assets/b0ed4a66-6881-40d5-a056-c755ecd51a0c)
```
 from sklearn.preprocessing import MaxAbsScaler
 scaler = MaxAbsScaler()
 df3=pd.read_csv("/content/bmi.csv")
 df3.head()
```
![image](https://github.com/user-attachments/assets/09b74579-051e-4156-8ddd-a64219f3e4b4)
```
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/bc82bd07-b664-4bc2-a6e6-fe696e7d681c)
```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("/content/bmi.csv")
df4.head()
```
![image](https://github.com/user-attachments/assets/6a7d4eee-2fff-40e5-bca2-1b635a03fb16)
```
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/user-attachments/assets/c2d2083c-5f32-4a55-83d6-496b2f7a198a)
```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![image](https://github.com/user-attachments/assets/a78c72ab-1e5b-47e1-b7e1-b8b9f3c5f392)
```
df
```
![image](https://github.com/user-attachments/assets/1de34510-ac62-4f94-925f-3d0812f41afe)
```
df.info()
```
![image](https://github.com/user-attachments/assets/313775bf-5641-469f-851e-0f21a8cebb45)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/31b9e792-04cb-4554-b422-d3e7d8c8e7e7)
```
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns]
```


# RESULT:
       # INCLUDE YOUR RESULT HERE
