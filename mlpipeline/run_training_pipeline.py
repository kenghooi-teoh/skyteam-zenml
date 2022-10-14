from pipelines.training_pipeline import training_pipeline
from steps.data_fetcher import fetch_train_data, fetch_val_data, fetch_label_data
from steps.data_preprocessor import training_data_preparation
from mlpipeline.steps.trainer import train_xgb_model
from steps.feature_engineer import feature_engineer_train, feature_engineer_val
from steps.model_evaluator import evaluator
from steps.deployment_trigger import deployment_trigger

from zenml.integrations.mlflow.steps import mlflow_model_deployer_step, MLFlowDeployerParameters


def run_training_pipeline():
    print("running training pipeline")
    deployer_params = MLFlowDeployerParameters(
        model_name="kh_model",
        experiment_name="kh_experiment",
        run_name="kh_run"
    )

    pipe = training_pipeline(
        fetch_train_data=fetch_train_data(),
        fetch_val_data=fetch_val_data(),
        fetch_label_data=fetch_label_data(),
        feature_engineer_train=feature_engineer_train(),
        feature_engineer_val=feature_engineer_val(),
        training_data_preparation=training_data_preparation(),
        train_xgb_model=train_xgb_model(),
        evaluate_model=evaluator(),
        deployment_trigger=deployment_trigger(),
        model_deployer=mlflow_model_deployer_step(params=deployer_params)
    )
    pipe.run()


if __name__ == "__main__":
    run_training_pipeline()
