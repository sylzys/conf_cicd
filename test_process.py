import pandas as pd
import pytest

from preprocess import aggregate_mean, calculate_charges


@pytest.fixture(scope='module')
def data():
    df = pd.DataFrame([[0, 2, 7, 4, 8],
                        [1, 7, 6, 3, 7],
                        [1, 1, "None", 8, 9],
                        [0, 2, 3, "None", 6],
                        [0, 5, 1, 4, 9]], 
                        columns = [f'feature_{i}' if i!=0 
                            else 'sex' for i in range(5)])
    return df


def test_calculate_charges():
    assert calculate_charges(1000, 2) == 250
   

@pytest.mark.parametrize("column, expected", [("feature_1", {0: 3, 1: 4}), ("feature_3", {0: 4, 1: 5.5})])   
def test_aggregate_mean_feature_1(data, column, expected):
    assert expected == aggregate_mean(data, column)

    