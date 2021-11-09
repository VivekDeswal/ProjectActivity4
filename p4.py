#Vivek Deswal
#180002

#IMPORTING LIBRARAIES

import pandas as pd # Pandas (version : 1.1.5)
import numpy as np # Numpy (version : 1.19.2)
import matplotlib.pyplot as plt # Matplotlib (version :  3.3.2)
from sklearn.cluster import KMeans # Scikit Learn (version : 0.23.2)
import seaborn as sns # Seaborn (version : 0.11.1)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import plotly as py
import plotly.graph_objs as go

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# EXPLORING THE DATASET

ds = pd.read_csv(r"F:\Semester 7\Mall_Customers.csv")
ds.head()
ds.tail()
ds.sample(10)
ds.columns
ds.describe()
ds.shape
ds.info()
ds.isnull().sum()


# STEP-3->DATA VISUALIZATION

# 1. Count plot of gender

sns.countplot(ds['Gender'])
sns.countplot(y = 'Gender', data = ds, palette="husl", hue = "Gender")
ds["Gender"].value_counts()

# 2. Plotting the relation b/w age, annual income and score

income = ds['Annual Income (k$)']
age = ds['Age']
score = ds['Spending Score (1-100)']

sns.lineplot(income, age, color = 'red')
sns.lineplot(income, score, color = 'blue')
plt.title('Annual Income versus Age and Spending Score', fontsize = 20)
plt.show()

sns.pairplot(ds, vars=["Age", "Annual Income (k$)", "Spending Score (1-100)"],  kind ="reg", palette="husl")

#  3. Distribution of values in age, annual income & spending score
#     according to Gender
sns.pairplot(ds, vars=["Age", "Annual Income (k$)", "Spending Score (1-100)"],  kind ="reg", hue = "Gender", palette="husl", markers = ['o','D'])


#STEP-4->CLUSTERING USING K MEANS

# 1. Segmentation using Age and Spending Score

sns.lmplot(x = "Age", y = "Spending Score (1-100)", data = ds, hue = "Gender")


# 2. Segmentation using Annual Income and Spending Score

sns.lmplot(x = "Annual Income (k$)", y = "Spending Score (1-100)", data = ds, hue = "Gender")


# 3. Segmentation using Age, Annual Income and Spending Score

sns.relplot(x="Annual Income (k$)", y="Spending Score (1-100)", size="Age", data=ds);


# STEP-5-> SELECTION OF CLUSTERS

X = ds.loc[:,["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
inertia = []
n = range(1,20)
for i in n:
    means_n = KMeans(n_clusters=i, random_state=0)
    means_n.fit(X)
    inertia.append(means_n.inertia_)

plt.plot(n , inertia , 'bo-')
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


# STEP-6-> PLOTTING THE CLUSTER BOUNDARY AND CLUSTER

ds.isnull().sum()
Y = ds.iloc[:, [3, 4]].values
ds = pd.get_dummies(ds, columns = ['Gender'], prefix = ['Gender'])

#Using KMeans for clustering
from sklearn.cluster import KMeans
a = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
    kmeans.fit(X)
    a.append(kmeans.inertia_)


# STEP-7->3D PLOT OF CLUSTERS

k_means = KMeans(n_clusters=5, random_state=0)
k_means.fit(X)
labels = k_means.labels_
centroids = k_means.cluster_centers_

trace = go.Scatter3d(
    x= X['Spending Score (1-100)'],
    y= X['Annual Income (k$)'],
    z= X['Age'],
    mode='markers',
     marker=dict(
        color = labels,
        size= 10,
        line=dict(
            color= labels,
        ),
        opacity = 0.9
     )
)
layout = go.Layout(
    title= 'Clusters',
    scene = dict(
            xaxis = dict(title  = 'Spending_score'),
            yaxis = dict(title  = 'Annual_income'),
            zaxis = dict(title  = 'Age')
        )
)
fig = go.Figure(data=trace, layout=layout)
fig.show()