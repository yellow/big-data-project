import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression,LogisticRegression
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def get_occupdata(a):
    df = pd.read_csv("nooutliers.TXT")
    df_cs = df[df["STEMOCCT"] == a]
    return df_cs

def scale_data(df_cs):
    X = df_cs[['SALARY', 'LOS']].values
    return X

def kmeans(X,k,df_cs):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    centroids = model.cluster_centers_
    sns.scatterplot(x='SALARY', y='LOS', data=df_cs,
                    s=60, hue=model.labels_, palette=['green', 'blue', 'orange', 'purple'])
    sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], marker='*',
                    s=400, color='red')
    plt.show()


