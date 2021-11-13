Codebase supporting my talk on CI/CD for MachineLearnia (Nov 12 2021)

The dataset used is available [here](https://www.kaggle.com/mirichoi0218/insurance/download).

The point of the talk is to demonstrate a simple end-to-end data project management, with code linting/quality, models training and monitoring, CI/CD and automatic deployment on Azure Webapp.

The tools used are:

- [MLFlow](https://www.mlflow.org/docs/latest/index.html) (Model training and monitoring, experiments and runs)
- [Poetry](https://python-poetry.org/) with following packages:

    - [flake8](https://flake8.pycqa.org/en/latest/) (pep8 standards)
    - [isort](https://pycqa.github.io/isort/) (imports sorting)
    - [pytest](https://docs.pytest.org/en/6.2.x/) (code testing)
    - [bandit](https://bandit.readthedocs.io/en/latest/) (security checking)
    - [safety](https://pypi.org/project/safety/) (at-risk libs checking)

- [Github Actions](https://github.com/features/actions) (CI / CD)
- [Flask](https://flask.palletsprojects.com/en/2.0.x/) (web app creation)
- [Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/) (Machine Learning environment)
