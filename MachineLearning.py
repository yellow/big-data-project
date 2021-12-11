import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import *
import seaborn as sns
from sklearn.metrics import classification_report
from scipy.stats import pointbiserialr, spearmanr
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



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
    df["LOC"] = df["LOC"].map({'DISTRICT OF COLUMBIA': 0, 'NEW YORK': 1, 'OHIO': 2, 'VIRGINIA': 3, 'GEORGIA': 4,
                               'FLORIDA': 5, 'COLORADO': 6, 'KANSAS': 7, 'PENNSYLVANIA': 8, 'NEW MEXICO': 9,
                               'NEVADA': 10, 'CALIFORNIA': 11, 'UTAH': 12, 'WASHINGTON': 13, 'TEXAS': 14,
                               'ARKANSAS': 15,'SOUTH DAKOTA': 16, 'ARIZONA': 17, 'DELAWARE': 18, 'MARYLAND': 19, 'MASSACHUSETTS': 20,
                               'NEBRASKA': 21, 'ILLINOIS': 22, 'OKLAHOMA': 23, 'LOUISIANA': 24, 'SOUTH CAROLINA': 25,
                               'MISSISSIPPI': 26, 'ALASKA': 27, 'NORTH DAKOTA': 28, 'ALABAMA': 29, 'HAWAII': 30,
                               'NEW JERSEY': 31, 'MONTANA': 32, 'TENNESSEE': 33, 'MISSOURI': 34, 'NORTH CAROLINA': 35,
                               'IOWA': 36, 'MAINE': 37, 'MINNESOTA': 38, 'MICHIGAN': 39, 'INDIANA': 40, 'OREGON': 41,
                               'WISCONSIN': 42, 'CONNECTICUT': 43, 'KENTUCKY': 44, 'IDAHO': 45, 'NEW HAMPSHIRE': 46,
                               'RHODE ISLAND': 47, 'WYOMING': 48, 'VERMONT': 49}).astype(int)
    return df


def logistic_regression(df):
    # Initialize the linear regression model
    df_x = pd.DataFrame(df)
    df_x = pd.DataFrame(np.c_[df['LOC'], df['AGELVL'], df['EDLVL'], df['LOS'], df['is_STEM']],
                        columns=["LOC", "AGELVL", "EDLVL", "LOS", "is_STEM"])
    df_y = pd.DataFrame(df.is_50k)
    reg = LogisticRegression()

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
    # Train our model with the training data
    reg.fit(x_train, np.ravel(y_train))
    # Initialize the linear regression model
    reg = LogisticRegression()

    reg.fit(x_train, np.ravel(y_train))
    y_pred = reg.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print('Classification Matrix:')
    print(classification_report(y_test, y_pred))
    return y_pred

def plot_importantfeatures(df):
    # decision tree for feature importance on a classification problem
    df_x = pd.DataFrame(df)
    df_x = pd.DataFrame(np.c_[df['LOC'], df['AGELVL'], df['EDLVL'], df['LOS'], df['is_STEM']],
                        columns=["LOC", "AGELVL", "EDLVL", "LOS", "is_STEM"])
    df_y = pd.DataFrame(df.is_50k)
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
    # define the model
    model = DecisionTreeClassifier()
    # fit the model
    model.fit(x_train, np.ravel(y_train))
    # get importance
    importance = model.feature_importances_
    labels = list(df_x)
    feature_importances = pd.DataFrame({'feature': labels, 'importance': importance})
    feature_importances = feature_importances[feature_importances.importance > 0.015]
    feature_importances.sort_values(by=['importance'], ascending=True, inplace=True)
    feature_importances['positive'] = feature_importances['importance'] > 0
    feature_importances.set_index('feature', inplace=True)
    feature_importances.importance.plot(kind='barh', figsize=(11, 6),
                                        color=feature_importances.positive.map({True: 'steelblue', False: 'red'}))
    plt.xlabel('Importance')
    plt.show()
