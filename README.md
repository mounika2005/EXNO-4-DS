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
![image](https://github.com/user-attachments/assets/edaaaf0f-8a9d-44ba-b624-71bcc248baa6)
```
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/ef6a9152-7094-4350-9670-3eea7921c7b3)
```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score
 from sklearn.ensemble import RandomForestClassifier
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 rf = RandomForestClassifier(n_estimators=100, random_state=42)
 rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/2754d89a-285c-4b30-9d02-c10fc0640c8c)
```
 y_pred = rf.predict(X_test)
 from sklearn.metrics import accuracy_score
 accuracy = accuracy_score(y_test, y_pred)
 print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/ca2e2421-65db-42ed-a087-15bf158c2621)
```
 import pandas as pd
 from sklearn.feature_selection import SelectKBest, chi2, f_classif
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/ff026ab1-4c7b-4f17-9e67-2e5a48540e5f)
```
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/96e17893-381c-4fa6-984c-e7eaa526601f)
```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 k_chi2 = 6
 selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
 X_chi2 = selector_chi2.fit_transform(X, y)
 selected_features_chi2 = X.columns[selector_chi2.get_support()]
 print("Selected features using chi-square test:")
 print(selected_features_chi2)
```
![image](https://github.com/user-attachments/assets/064d58d0-0c2f-43bf-a5c2-f4a58b1367c9)
```
 selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
       'hoursperweek']
 X = df[selected_features]
 y = df['SalStat']
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 rf = RandomForestClassifier(n_estimators=100, random_state=42)
 rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/1f52319b-dae3-4316-bb30-54056056a9ef)

```
 y_pred = rf.predict(X_test)
 from sklearn.metrics import accuracy_score
 accuracy = accuracy_score(y_test, y_pred)
 print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/08c438c8-c400-4ef9-beb5-3fb330aa960a)
```
# @title
!pip install skfeature-chappers
```
![image](https://github.com/user-attachments/assets/220016c7-34e4-44c2-ae5b-1f069d5ec920)
```
 # @title
 import numpy as np
 import pandas as pd
 from skfeature.function.similarity_based import fisher_score
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score
 # @title
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 # @title
 df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/330b0aff-c3db-460c-a0ad-bea1e8629ef3)
```
 # @title
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 # @title
 df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/373b8ea2-ef20-4bd6-af6b-500c9d6d7f24)
```
 # @title
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
```
```
from sklearn.feature_selection import SelectKBest, f_classif 
# import SelectKBest and f_classif
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![image](https://github.com/user-attachments/assets/7e92506d-d905-4791-8aaf-2993c12cd288)
```
cimport pandas as pd
 from sklearn.feature_selection import RFE
 from sklearn.linear_model import LogisticRegression
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/9ae5b89c-dd16-4e51-9e56-35f81838bfa7)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/7f818606-ca70-4b0e-8615-07035c9aba58)
```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 logreg = LogisticRegression()
 n_features_to_select = 6
 rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
 rfe.fit(X, y)
```
![image](https://github.com/user-attachments/assets/375e031e-daa4-409f-b67f-75061b9e0d1c)
```
 selected_features = X.columns[rfe.support_]
 print("Selected features using RFE:")
 print(selected_features)
```
![image](https://github.com/user-attachments/assets/dee7434e-55a2-4872-b8fa-b00c18858021)
```
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score
 from sklearn.ensemble import RandomForestClassifier
 X_selected = X[selected_features]
 X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
 rf = RandomForestClassifier(n_estimators=100, random_state=42)
 rf.fit(X_train, y_train)
 y_pred = rf.predict(X_test)
 accuracy = accuracy_score(y_test, y_pred)
 print(f"Model accuracy using Fisher Score selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/94a0651f-5cf5-42e6-b5de-d430a6c1f4f8)

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
