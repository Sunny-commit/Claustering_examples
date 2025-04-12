# -*- coding: utf-8 -*-
"""Clausting_Examples_1

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BPsqJPWam4aZysQ7gsqcfJFvHUF_49gC
"""

! pip install -q kaggle

pip install scikit-learn-extra

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('/content/segmentation_data[1].csv')
df

df.isnull().sum()

for i in df.columns:
  sns.distplot(df[i])
  plt.show()

df.info()

"""##Heatmap"""

plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),cmap='RdBu',vmax=1,vmin=-1,annot=True)
plt.savefig('heatmap.png')
plt.show()

sns.distplot(df['Age'], bins = 30)

plt.figure(figsize=(5,5))
plt.boxplot(x='Age',y='Income',data=df,palette='pastel')
plt.show()

"""Training using clustering"""

X=df.drop('ID',axis=1)
X=df.iloc[:,[2,4]].values

from sklearn.cluster import KMeans
distortion=[]
for i in range(1,20):
  k=KMeans(n_init=1,n_clusters=i,init='k-means++',random_state=42)
  k.fit(X)
  distortion.append(k.inertia_)
plt.plot(range(1,20),distortion,marker='o')
plt.xlabel('No of cluster')
plt.ylabel('Distortion')
plt.show()

"""##Using best case"""

model=KMeans(n_clusters=4,init='k-means++',random_state=42)
model.fit(X)
y=model.fit_predict(X)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

#X=sc.inverse_transform(X)

plt.scatter(X[y==0,0],X[y==0,1],s=50,c='red',label='Cluster1')
plt.scatter(X[y==1,0],X[y==1,1],s=50,c='green',label='Cluster2')
plt.scatter(X[y==2,0],X[y==2,1],s=50,c='pink',label='Cluster3')
plt.scatter(X[y==3,0],X[y==3,1],s=50,c='blue',label='Cluster4')
plt.scatter(X[y==4,0],X[y==4,1],s=50,c='violet',label='Cluster5')
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=100,c='yellow',label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()