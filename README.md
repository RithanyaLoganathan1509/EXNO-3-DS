## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
# Encoding:
```
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('data.csv')
print(df)
```
<img width="622" height="247" alt="image" src="https://github.com/user-attachments/assets/d6b35133-5cc6-40b4-8ed0-294dc827dbc9" />

# Ordinal Encoding:
```
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
climate = ['Cold','Warm','Hot','Very Hot']
ele = OrdinalEncoder(categories=[climate])
ele.fit_transform(df[["Ord_1"]])
```
<img width="198" height="230" alt="image" src="https://github.com/user-attachments/assets/4d0c4df7-edf8-4ecd-baf5-8855ba9c7884" />

```
df['bo2'] = ele.fit_transform(df[["Ord_1"]])
df
```
<img width="536" height="369" alt="image" src="https://github.com/user-attachments/assets/d56ed7b6-9051-49ec-bd86-c3a300360a04" />

# Label Encoding:
```
le = LabelEncoder()
df2 = df.copy()
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
```
<img width="492" height="359" alt="image" src="https://github.com/user-attachments/assets/1e5a2a13-7ffd-40db-bbe5-a24a50e5bcd3" />

```
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
```
<img width="488" height="356" alt="image" src="https://github.com/user-attachments/assets/dcb7bbf1-a6a6-4c3a-9738-7b067c0adf03" />

# OneHot Encoding:
```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
df3 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["City"]]))
df2 = pd.concat([enc,df3],axis = 1)
df2
```
<img width="597" height="346" alt="image" src="https://github.com/user-attachments/assets/de28d684-23e1-4ca5-8687-407f5b10796c" />

```
pd.get_dummies(df,columns=['City'])
```
<img width="861" height="357" alt="image" src="https://github.com/user-attachments/assets/39de7716-0240-48d7-b192-606d2002cb97" />

# Binary Encoding:
```
from category_encoders import BinaryEncoder
df = pd.read_csv('data.csv')
df
```
<img width="475" height="346" alt="image" src="https://github.com/user-attachments/assets/36f87ba2-0c34-4ee1-b89c-be624193c338" />

```
be = BinaryEncoder()
nd = be.fit_transform(df['Ord_2'])
df
```
<img width="483" height="360" alt="image" src="https://github.com/user-attachments/assets/ea7f7513-6e93-4585-a642-b9df600598dd" />

# Target Encoding:
```
from category_encoders import TargetEncoder
te = TargetEncoder()
CC = df.copy()
new = te.fit_transform(CC["City"],y=CC["Target"])
CC = pd.concat([CC,new],axis = 1)
CC
```
<img width="563" height="366" alt="image" src="https://github.com/user-attachments/assets/4df96b76-2140-4b30-bf00-5ef7a6892e88" />

```
if 'City' in CC.columns:
    CC = CC.drop('City', axis=1)
new = te.fit_transform(X = df["City"],y=df["Target"])
CC = pd.concat([CC.reset_index(drop=True),new.reset_index(drop=True)],axis = 1)
CC
```
<img width="476" height="364" alt="image" src="https://github.com/user-attachments/assets/c3a8ba24-2a32-40c8-845d-6caabb7e94dc" />

# Transformation:
```
df = pd.read_csv('Data_to_Transform.csv')
df
```
<img width="771" height="429" alt="image" src="https://github.com/user-attachments/assets/7d51319f-85da-464c-9cfe-d2f1dda992f8" />

```
df.skew()
```
<img width="396" height="118" alt="image" src="https://github.com/user-attachments/assets/895647f9-a40b-47a9-9e93-722627fd7b03" />

# Function Transformation:
```
import numpy as np
np.log(df["Highly Positive Skew"])
```
<img width="653" height="273" alt="image" src="https://github.com/user-attachments/assets/62e4a506-21cf-4414-94f1-79fa94700b1d" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="652" height="271" alt="image" src="https://github.com/user-attachments/assets/187621e0-ae98-4697-bc3b-add606576ae5" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="653" height="273" alt="image" src="https://github.com/user-attachments/assets/cf11382c-1713-491c-acce-3a7cd50c36a4" />

```
np.square(df["Highly Positive Skew"])
```
<img width="733" height="267" alt="image" src="https://github.com/user-attachments/assets/ee5e4c39-7115-414c-a451-5a8f7260be79" />

# Power Transformation:
```
from scipy import stats
df["Highly Positive Skew_boxcox"], parameters = stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1048" height="440" alt="image" src="https://github.com/user-attachments/assets/f8feb67f-8198-4b4f-829f-3895ab85a247" />

```
df["Moderate Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Moderate Negative Skew"])
df
```
<img width="1230" height="437" alt="image" src="https://github.com/user-attachments/assets/bc7b84b8-0d19-4055-82cf-cba3939ce4a4" />

```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1239" height="432" alt="image" src="https://github.com/user-attachments/assets/115fcf98-39fb-4817-91ec-d6be0cf36dd7" />

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line = '45')
plt.show()
```
<img width="868" height="528" alt="image" src="https://github.com/user-attachments/assets/4918dabb-9453-4e1d-afc1-7c938905a427" />

```
sm.qqplot(df["Moderate Negative Skew_1"],line = '45')
plt.show()
```
<img width="739" height="522" alt="image" src="https://github.com/user-attachments/assets/795e9ca0-5531-4e5c-9540-d5e1a1fae165" />

```
df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line = '45')
plt.show()
```
<img width="760" height="542" alt="image" src="https://github.com/user-attachments/assets/f756c499-e73c-4843-ab35-9feb42f25acc" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line = '45')
plt.show()
```
<img width="881" height="528" alt="image" src="https://github.com/user-attachments/assets/672346bd-f009-487a-9114-3e6f673b4452" />

```
sm.qqplot(df["Highly Negative Skew_1"],line = '45')
plt.show()
```
<img width="817" height="538" alt="image" src="https://github.com/user-attachments/assets/9c158b5c-4020-476b-b240-c9a23843cbd0" />

```
sm.qqplot(np.abs(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```
<img width="762" height="533" alt="image" src="https://github.com/user-attachments/assets/e0cfe630-8622-44f5-a79c-06d101bf8cc9" />

```
sm.qqplot(np.log(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```
<img width="776" height="540" alt="image" src="https://github.com/user-attachments/assets/d7b02eb3-1745-4ec2-a3e0-e61e57f443f2" />

```
sm.qqplot(np.sqrt(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
<img width="856" height="551" alt="image" src="https://github.com/user-attachments/assets/23fcea75-10d8-438d-96ef-63fa3190eaeb" />

```
pd.concat([CC,new],axis = 1)
```
<img width="568" height="365" alt="image" src="https://github.com/user-attachments/assets/1df27b44-39b8-4e07-8d70-8949434d2014" />

# RESULT:
Thus, we have successfully performed Feature Encoding and Transformation process and saved the data to a file.

       
