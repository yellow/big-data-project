import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier




def load_data():
    df = pd.read_csv('nooutliers.TXT', index_col=0)
    return df

def transform_data(df):
    # dealing with categorical data
    df = df[["LOC", "is_50k", "AGELVL", "EDLVL", "LOS", "is_STEM"]]
    df["AGELVL"] = df["AGELVL"].map(
        {"0-20": 0, "20-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6, "50-54": 7, "55-59": 8,
         "60-64": 9, "65-120": 10}).astype(int)
    df["EDLVL"] = df["EDLVL"].map(
        {'BACHELORS': 4, 'POST-BACHELORS': 5, 'DOCTORATE': 8, 'BELOW HIGH SCHOOL': 0, 'MASTERS': 6,
         'BETWEEN HS & BACHELORS': 3, 'HIGH SCHOOL OR EQUIVALENCY': 1, 'POST MASTERS': 7, 'OCCUPATIONAL PROGRAM': 2,
         'POST-DOCTORATE': 9}).astype(int)

    labelenc = LabelEncoder()
    df["LOC_N"] = labelenc.fit_transform(df["LOC"])

    return df



def plot_importantfeatures(df):
    # decision tree for feature importance on a classification problem
    df_x = pd.DataFrame(np.c_[df['LOC_N'], df['AGELVL'], df['EDLVL'], df['LOS'], df['is_STEM']],
                        columns=["LOC_N", "AGELVL", "EDLVL", "LOS", "is_STEM"])
    df_y = pd.DataFrame(df.is_50k)
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
    # define the model
    model = DecisionTreeClassifier()
    # fit the model
    model.fit(x_train, np.ravel(y_train))
    # get importance
    y_pred = model.predict(x_test)
    importance = model.feature_importances_
    labels = list(df_x)
    feature_importances = pd.DataFrame({'feature': labels, 'importance': importance})
    feature_importances = feature_importances[feature_importances.importance > 0.015]
    feature_importances.sort_values(by=['importance'], ascending=True, inplace=True)
    feature_importances['positive'] = feature_importances['importance'] > 0
    feature_importances.set_index('feature', inplace=True)
    feature_importances.importance.plot(kind='barh', figsize=(11, 6),
                                        color=feature_importances.positive.map({True: 'steelblue', False: 'red'}))
    print('Classification Matrix:')
    print(classification_report(y_test, y_pred))
    plt.xlabel('Importance')
    plt.show()
