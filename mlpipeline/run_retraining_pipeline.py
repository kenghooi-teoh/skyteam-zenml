from datetime import datetime

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from mlpipeline.steps.prediction_service_loader import PredictionServiceLoaderStepConfig, prediction_service_loader
from mlpipeline.steps.trainer import train_xgb_model
from pipelines.training_pipeline import training_pipeline
from steps.data_fetcher import fetch_train_data, fetch_val_data, fetch_label_data, FetchDataConfig
from steps.data_preprocessor import training_data_preparation
from steps.feature_engineer import feature_engineer_train, feature_engineer_val
from steps.model_evaluator import evaluator
from steps.training_config import training_config, TrainingConfig


def run_retraining_pipeline():
    config = TrainingConfig(is_retraining=True)

    train_start_date = datetime(2022, 6, 1)
    train_end_date = datetime(2022, 6, 30)

    val_start_date = datetime(2022, 7, 1)
    val_end_date = datetime(2022, 7, 31)

    fetch_train_data_config = FetchDataConfig(start_date=str(train_start_date.date()), end_date=str(train_end_date.date()))
    fetch_val_data_config = FetchDataConfig(start_date=str(val_start_date.date()), end_date=str(val_end_date.date()))

    predictor_service_config = PredictionServiceLoaderStepConfig(
        pipeline_name="training_pipeline",
        step_name="model_deployer",
        model_name="xgboost"
    )

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
        prediction_service_loader=prediction_service_loader(config=predictor_service_config),
        model_deployer=mlflow_model_deployer_step()
    )
    pipe.run()


if __name__ == "__main__":
    run_retraining_pipeline()
