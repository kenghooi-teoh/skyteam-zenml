from steps.training_config import training_config, TrainingConfig
from pipelines.training_pipeline import training_pipeline
from steps.data_fetcher import fetch_train_data, fetch_val_data, fetch_label_data, FetchDataConfig
from steps.data_preprocessor import training_data_preparation
from mlpipeline.steps.trainer import train_xgb_model
from steps.feature_engineer import feature_engineer_train, feature_engineer_val
from steps.model_evaluator import evaluator

from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

from datetime import datetime

def run_retraining_pipeline():
    print("running pipeline")

    print("MLflow tracking URI:", get_tracking_uri())

    config = TrainingConfig(is_retraining=True)

    train_start_date = datetime(2022, 6, 1)
    train_end_date = datetime(2022, 6, 30)

    val_start_date = datetime(2022, 7, 1)
    val_end_date = datetime(2022, 7, 31)

    fetch_train_data_config = FetchDataConfig(start_date=str(train_start_date.date()), end_date=str(train_end_date.date()))
    fetch_val_data_config = FetchDataConfig(start_date=str(val_start_date.date()), end_date=str(val_end_date.date()))

    pipe = training_pipeline(
        training_config=training_config(config=config),
        fetch_train_data=fetch_train_data(config=fetch_train_data_config),
        fetch_val_data=fetch_val_data(config=fetch_val_data_config),
        fetch_label_data=fetch_label_data(),
        feature_engineer_train=feature_engineer_train(),
        feature_engineer_val=feature_engineer_val(),
        training_data_preparation=training_data_preparation(),
        train_xgb_model=train_xgb_model(),
        evaluate_model=evaluator(),
        model_deployer=mlflow_model_deployer_step()
    )
    pipe.run()


if __name__ == "__main__":
    run_retraining_pipeline()