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

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")

df.head()
df.dropna()
```

# Output


![image](https://github.com/user-attachments/assets/26796ef2-3005-45ce-95a6-b106fd392acb)

```
max_vals=np.max(np.abs(df[['Height']]))
max_vals
max_vals1=np.max(np.abs(df[['Weight']]))
max_vals1
print("Height =",max_vals)
print("Weight =",max_vals1)
```

# Output

![image](https://github.com/user-attachments/assets/55dbc3d0-9865-45eb-8a87-023f05fe616c)


```
df1=pd.read_csv("/content/bmi.csv")

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```

# Output

![image](https://github.com/user-attachments/assets/5bcec15b-1a05-4958-a4f1-53ef7b25423f)


```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

# Output

![image](https://github.com/user-attachments/assets/602e8dbc-ff3e-498d-90f5-726c90685167)

```
from sklearn.preprocessing import Normalizer
df2=pd.read_csv("/content/bmi.csv")
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

# Output

![image](https://github.com/user-attachments/assets/fcecfdaf-5b19-4077-8f8c-4861eb615e57)


```
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```

# Output

![image](https://github.com/user-attachments/assets/2bb6e3b5-9ba6-4f33-9bd4-2baf211909ea)


```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```

# Output

![image](https://github.com/user-attachments/assets/64a7b2b4-aa65-4c0e-9443-0755f84e5459)


## FEATURE SELECTION SUING KNN CLASSIFICATION

```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income.csv",na_values=[ " ?"])
data
```

# Output

![image](https://github.com/user-attachments/assets/b4671f97-d63b-4939-ab74-c4273929b6b1)

```
data.isnull().sum()
```

# Output

![image](https://github.com/user-attachments/assets/bdeae9ea-bd7e-4848-8a31-5e01828a4b53)


```
missing=data[data.isnull().any(axis=1)]
missing
```

# Output

![image](https://github.com/user-attachments/assets/29252d99-a1fc-4f83-a02d-8ae603ff71f3)


```
data2=data.dropna(axis=0)
data2
```

# Output

![image](https://github.com/user-attachments/assets/7c692a8d-0899-4aa0-b647-ac678147ddf8)


```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

# Output

![image](https://github.com/user-attachments/assets/d972eb04-43fa-4f48-82f2-b1e1168c42e2)


```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```

# Output

![image](https://github.com/user-attachments/assets/8664b65a-b49b-489f-b9df-c7142f509ffa)


```
data2
```

# Output

![image](https://github.com/user-attachments/assets/6a8851da-2386-40be-9896-b1814ce88d1e)


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

# Output

![image](https://github.com/user-attachments/assets/816396fa-8774-4688-a3b9-4c33af1c2be7)


```
columns_list=list(new_data.columns)
print(columns_list)
```

# Output

![image](https://github.com/user-attachments/assets/a3cb21ed-bdc9-4c35-834d-b82e966ee336)


```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

# Output

![image](https://github.com/user-attachments/assets/087ba919-bedc-4de6-b794-800c7d2e024a)


```
y=new_data['SalStat']
print(y)
```

# Output

![image](https://github.com/user-attachments/assets/97a3d74f-51e0-4a1b-81c1-3a451c8ad990)

```
x=new_data[features].values
print(x)
```

# Output

![image](https://github.com/user-attachments/assets/7f95e089-b012-4100-832b-62e608404dfe)


```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

# Output

![image](https://github.com/user-attachments/assets/89ab8fca-6aee-4dee-a018-1e92d8b9a3b1)


```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

# Output

![image](https://github.com/user-attachments/assets/a5203713-c834-4b87-b546-cf02e4caf886)


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data= {
    'Feature1' : [1,2,3,4,5],
    'Feature2' : ['A','B','C','A','B'],
    'Feature3' : [0,1,1,0,1],
    'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)

X=df[['Feature1','Feature3']]
y=df['Target']

selector = SelectKBest(score_func= mutual_info_classif, k=1)
X_new=selector.fit_transform(X,y)

selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]

print("Selected Features:", selected_features)

```

# Output

![image](https://github.com/user-attachments/assets/7aa7af4a-0334-45d7-81d1-713119c3e491)


# RESULT:
   Thus, Feature selection and Feature scaling has been used on the given dataset.
