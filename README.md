# Mini-Project
# Weather-analysis.
# Aim : Analysis Of Weather In The Data Science.
## Procedure:








# Program And Output:
# Importing necessary packages:
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
~~~

# read the data set:
~~~
df=pd.read_csv("weather.csv")
df
~~~
![image](https://user-images.githubusercontent.com/94226297/202074920-223a336a-361e-440b-b31e-7cca3a1c8a53.png)
~~~
df.head()
~~~
![image](https://user-images.githubusercontent.com/94226297/202075017-6ab7c3b0-cbe9-4f3e-9f6e-56d3fcdcd8be.png)
~~~

df.info()
~~~
![image](https://user-images.githubusercontent.com/94226297/202075224-10709e9c-0f94-4a01-b056-47661fad990b.png)
~~~
df.tail()
~~~
![image](https://user-images.githubusercontent.com/94226297/202075270-aeb27ea2-cefb-45c4-a909-a60d8a9bb336.png)
~~~
df.describe()
~~~
![image](https://user-images.githubusercontent.com/94226297/202075365-f97cbbe5-31f6-4cff-9fd9-ebff97e80e50.png)
~~~
df.shape
~~~
![image](https://user-images.githubusercontent.com/94226297/202075419-7e9ba26a-104f-45bb-b05e-04ca0eb3bdab.png)
~~~
df['weather'].value_counts()
~~~
![image](https://user-images.githubusercontent.com/94226297/202075488-ee9c5852-bb17-45b0-984a-ecdc225e558d.png)

# label encoder:
~~~
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['wind'] = le.fit_transform(df['weather'])
df.head(10)
~~~
![image](https://user-images.githubusercontent.com/94226297/202075544-2ff2b185-77a9-4c4e-a563-9ae28f9cbfc0.png)

# data cleaning:
~~~
df.isnull().sum()
~~~
![image](https://user-images.githubusercontent.com/94226297/202075615-c47bf868-a9e7-4c67-8730-0cabf28a9031.png)
~~~
missing_percentage = (df.isnull().sum())/(df.shape[0])*100
missing_percentage
~~~
![image](https://user-images.githubusercontent.com/94226297/202075672-562ded93-db62-40c9-82b7-22ffd1cbb391.png)
~~~
df.duplicated().value_counts()
~~~
![image](https://user-images.githubusercontent.com/94226297/202075848-abb5493b-bd9f-4f0c-9cc8-9be3065e9787.png)

# Univariate Analysis:
~~~
sns.boxplot(y="wind",data=df)
~~~
![image](https://user-images.githubusercontent.com/94226297/202076084-b5248e03-6180-4c82-8a94-a33c47486781.png)
~~~
sns.countplot(y="weather",data=df)
~~~
![image](https://user-images.githubusercontent.com/94226297/202076467-57071e75-3cc2-42f9-8c88-6d3fdcb09e57.png)
~~~
sns.histplot(y="wind",data=df)
~~~
![image](https://user-images.githubusercontent.com/94226297/202076570-7d0e696e-32eb-44a4-adc5-f50e0a78beff.png)

# Multivariate Analysis:

~~~
sns.scatterplot(df['wind'],df['weather'])
~~~

![image](https://user-images.githubusercontent.com/94226297/202077081-7637e653-4d25-4849-b2d0-3c2ebcbd90f9.png)

~~~
sns.barplot(data=df, x='wind', y='precipitation')
~~~
![image](https://user-images.githubusercontent.com/94226297/202077186-363af693-a67a-490c-8d23-ed695fdf3330.png)

~~~
df.corr()
~~~
![image](https://user-images.githubusercontent.com/94226297/202077255-46a8858c-e616-4b90-84ab-d983e569436f.png)

~~~
sns.heatmap(df.corr(),annot=True)
~~~
![image](https://user-images.githubusercontent.com/94226297/202077324-79db0496-ba14-41c2-afaf-4c1fddaab9de.png)

# Data Visualization:
~~~
plt.figure(figsize=(20, 7))
sns.lineplot(data=df, x='temp_min', y='temp_max')
plt.show()
~~~
![image](https://user-images.githubusercontent.com/94226297/202077469-d20438b6-2b8e-4ee0-b845-0fee37ea002d.png)

~~~
sns.pointplot(x=df['temp_max'],y=df['temp_min'])
~~~
![image](https://user-images.githubusercontent.com/94226297/202077572-280474f5-d721-4857-8009-82959be22820.png)

~~~
sns.kdeplot(x=df['wind'],data=df)
~~~
![image](https://user-images.githubusercontent.com/94226297/202077649-689749e0-9b44-43de-9f90-48e45724fb2e.png)
~~~
sns.countplot(y="precipitation",data=df)
~~~
![image](https://user-images.githubusercontent.com/94226297/202077705-a325ad73-9ac3-4ca4-a694-607793f5e798.png)
