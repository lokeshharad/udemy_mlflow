import warnings
import argparse
import logging
from logging import warning
from operator import index

import numpy as np
import pandas as pd
from graphql import print_type
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from pathlib import Path

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)
args = parser.parse_args()

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = np.sqrt(mean_absolute_error(actual, pred))
    r2 = np.sqrt(r2_score(actual, pred))
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data = pd.read_csv("../datasets/red-wine-quality.csv")
    # print(data.head())

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    train_x = train_data.drop(["quality"], axis=1)
    train_y = train_data["quality"]
    test_x = test_data.drop(["quality"], axis=1)
    test_y = test_data["quality"]
    print("data: ", data.shape)
    print("train data: ", train_x.shape, train_y.shape)
    print("test data: ", test_x.shape, test_y.shape)

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri("")
    print("The set tracking uri is ", mlflow.get_tracking_uri())
    exp= mlflow.set_experiment("experiment_2")
    # print(exp_id)

    # get_exp = mlflow.get_experiment(exp_id)
    print("Name: ", exp.name)
    print("Exp ID: ", exp.experiment_id)
    print("Artifact Location: ", exp.artifact_location)
    print("Tags: ", exp.tags)
    print("LifeCycle Stage: ", exp.lifecycle_stage)
    print("Creation Time: ", exp.creation_time)
    mlflow.start_run(experiment_id=exp.experiment_id, run_name="run_2")
    # mlflow.start_run(run_id="624949e90cf14c958408a2bc884b9782")

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    pred_qualities = lr.predict(test_x)

    rmse, mae, r2 = eval_metrics(test_y, pred_qualities)
    print("ElasticNet Model(alpha={alpha}, l1_ratio={l1_ratio})".format(alpha=alpha, l1_ratio=l1_ratio))
    print("rmse: ", rmse)
    print("mae: ", mae)
    print("r2: ", r2)

    params = {
        "alpha": alpha,
        "l1_ratio": l1_ratio
    }
    metrics_d = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    mlflow.log_params(params)
    mlflow.log_metrics(metrics_d)

    mlflow.sklearn.log_model(lr, "wine_quality_new_model_2")

    run_obj = mlflow.active_run()
    print("Current Active run id: ", run_obj.info.run_id)
    print("Current Active run name: ", run_obj.info.run_name)

    mlflow.end_run()

    last_run_obj = mlflow.last_active_run()
    print("Active run id: ", last_run_obj.info.run_id)
    print("Active run name: ", last_run_obj.info.run_name)