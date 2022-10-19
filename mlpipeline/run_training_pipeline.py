from mlpipeline.steps.prediction_service_loader import PredictionServiceLoaderStepConfig, prediction_service_loader
from steps.training_config import training_config, TrainingConfig
from pipelines.training_pipeline import training_pipeline
from steps.data_fetcher import fetch_train_data, fetch_val_data, fetch_label_data
from steps.data_preprocessor import training_data_preparation
from mlpipeline.steps.trainer import train_xgb_model
from steps.feature_engineer import feature_engineer_train, feature_engineer_val
from steps.model_evaluator import evaluator

from zenml.integrations.mlflow.steps import mlflow_model_deployer_step


from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

def run_training_pipeline():
    print("running pipeline")

    print("MLflow tracking URI:", get_tracking_uri())

    config = TrainingConfig(is_retraining=False)

    predictor_service_config = PredictionServiceLoaderStepConfig(
        pipeline_name="training_pipeline",
        step_name="model_deployer",
        model_name="xgboost"
    )

    pipe = training_pipeline(
        training_config=training_config(config=config),
        fetch_train_data=fetch_train_data(),
        fetch_val_data=fetch_val_data(),
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
    run_training_pipeline()