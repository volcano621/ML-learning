import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def set_missing_ages(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    y = known_age[:, 0]
    x = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    # 表示使用2000个决策树，n_jobs表示使用cpu的核数，random_state表示随机种子
    rfr.fit(x, y)
    predicted_ages = rfr.predict(unknown_age[:, 1::])  # 1::表示从第一个元素开始，每隔一个元素取一个元素
    df.loc[(df.Age.isnull()), 'Age'] = predicted_ages
    return rfr


def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'


def one_hot_encoder(df, col):
    one_hot = OneHotEncoder(sparse_output=False)
    df_encoded = pd.DataFrame(one_hot.fit_transform(df[col]))
    df_encoded.columns = one_hot.get_feature_names_out()
    df.drop(col, axis=1, inplace=True)
    df = pd.concat([df, df_encoded], axis=1)
    return df


def standard_scaler(df, col):
    scaler = StandardScaler()
    df[col] = scaler.fit_transform(df[col])





