
from datetime import datetime

import bios
import explainerdashboard as expdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
from azureml.core import Model, Workspace
from explainerdashboard import (ExplainerDashboard, InlineExplainer,
                                RegressionExplainer)
from explainerdashboard.custom import *
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, train_test_split)

import mlflow

config = bios.read("config.yaml")
df = joblib.load(config["dumps"] + "/df")


def create_artifact(func, filename):
    path = config["dumps"] + filename
    func.write_image(path)
    mlflow.log_artifact(path)

X = df.drop("charges", 1)
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# mlflow.set_tracking_uri("./mlruns")

'''
Uncomment the 2 following lines to push experiments to Azure
'''
# ws = Workspace.from_config("config.json")
# mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

'''
Uncomment the following line to push experiments to local mysql with command:
mlflow server --backend-store-uri   mysql+pymysql://root@localhost/mlflow --default-artifact-root file:/./mlruns -h 0.0.0.0 -p 8000
'''
# ws = Workspace.from_config("http://localhost:8000")

mlflow.set_experiment(config["mlflow"]["experiment_name"])

for model_name in config['models']:
    with mlflow.start_run() as run:
        model = joblib.load(config["dumps"] + model_name)
        param_grid = dict(config["grid_params"][model_name])

        for key, value in param_grid.items():
            if isinstance(value, str):
                param_grid[key] = eval(param_grid[key])

        param_grid = {element: (list(param_grid[element])) for element in param_grid}
        grid_search = GridSearchCV(
            model,
            param_grid,
            scoring=config["GridSearchCV"]["scoring"],
            cv=config["GridSearchCV"]["cv"],
            n_jobs=config["GridSearchCV"]["n_jobs"]
        )

        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        y_pred = model.predict(X_test)

        mlflow.sklearn.log_model(grid_search, model_name)

        mlflow.sklearn.save_model(model, "models/insurance-{}-{}".format(
            model_name, datetime.now().strftime("%d_%m_%Y-%H-%M")
        ))
        joblib.dump(grid_search.best_estimator_, "best_model")

        explainer = RegressionExplainer(model, X_test, y_test, target='charges')

        create_artifact(explainer.plot_predicted_vs_actual(), 'predvsactual.png')
        mlflow.log_artifact(config["dumps"] + 'predvsactual.png')

        create_artifact(explainer.plot_importances(kind='shap', topx=5, round=3), 'feat_imp.png')
        mlflow.log_artifact(config["dumps"] + 'feat_imp.png')

        create_artifact(explainer.plot_residuals_vs_feature("bmi"), 'residual_vs_bmi.png')
        mlflow.log_artifact(config["dumps"] + 'residual_vs_bmi.png')

        for name, value in explainer.metrics().items():
            mlflow.log_metric(name, value)
