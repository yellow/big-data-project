import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def get_occupdata(a):
    df = pd.read_csv("nooutliers.TXT")
    df_cs = df[df["STEMOCCT"] == a]
    return df_cs

def scale_data(df_cs):
    frequency_sqrt = np.sqrt(df_cs['LOS'])
    df_cs["LOS"] = frequency_sqrt
    X = df_cs[['SALARY', 'LOS']].values
    scaler = StandardScaler()
    scaler.fit(X)
    normalized_data = scaler.transform(X)
    return X

def kmeans(X,k,df_cs):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    centroids = model.cluster_centers_
    sns.scatterplot(x='SALARY', y='LOS', data=df_cs,
                    s=60, hue=model.labels_)
    clusters = df_cs
    clusters["Cluster"] = model.labels_
    print(clusters[["LOC", "AGELVL", "EDLVL", "STEMOCC", "SALARY", "LOS", "GEN", "Cluster"]])
    plt.show()


