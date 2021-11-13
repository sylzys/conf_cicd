import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.model_selection import (GridSearchCV, train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

OUTPUT = "dumps"
INPUT = "input"

df = pd.read_csv(INPUT+'/insurance.csv')
print(df.head())

X = df.drop("charges", 1)
y = df["charges"]

def aggregate_mean(df, col):
    return df.groupby("sex")[col].mean().to_dict()

def calculate_charges(charges, children):
    return(charges // (children + 2))

# "FEATURE ENGINEERING"
# df['mean_charge_per_family_member'] = calculate_charges(df['charges'], df['children'])
# print(df.head())

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)
joblib.dump(df, OUTPUT+"/df")

# PIPELINES

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'bool']).columns

categorical_transformer = Pipeline(
    steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)),
        ('scaler', StandardScaler())
    ]
)

numeric_transformer = Pipeline(
    steps=[
        ('scaler', StandardScaler())
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ]
)

lasso = make_pipeline(
        preprocessor,
        Lasso(max_iter = 1000)
    )

svr =  make_pipeline(
        preprocessor,
        SVR()
    )

joblib.dump(lasso, "dumps/Lasso")
joblib.dump(svr, "dumps/SVR")