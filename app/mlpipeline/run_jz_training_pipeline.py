import pandas as pd
import xgboost as xgb
from zenml.integrations.mlflow.steps import (
    mlflow_model_deployer_step
)
from zenml.pipelines import pipeline
from zenml.steps import step, Output, BaseParameters

from mlpipeline.steps.data_fetcher import fetch_train_data, fetch_val_data, fetch_label_data, FetchDataConfig
from mlpipeline.steps.data_preprocessor import training_data_preparation
from mlpipeline.steps.feature_engineer import feature_engineer_train, feature_engineer_val
from mlpipeline.steps.prediction_service_loader import PredictionServiceLoaderStepConfig
from mlpipeline.steps.trainer import train_xgb_model
from mlpipeline.steps.util import amex_metric_mod


class EvaluatorStepConfig(BaseParameters):
    """Model deployment service loader configuration
    Attributes:
        pipeline_name: name of the pipeline that deployed the model prediction
            server
        step_name: the name of the step that deployed the model prediction
            server
        model_name: the name of the model that was deployed
    """

    pipeline_name: str
    step_name: str
    model_name: str


@step(enable_cache=False)
def evaluator(model: xgb.core.Booster, x_val: pd.DataFrame, y_val:pd.Series) -> Output(accuracy=float, deployment_decision=bool):
    valid_dmatrix = xgb.DMatrix(data=x_val, label=y_val)

    oof_preds = model.predict(valid_dmatrix)
    accuracy = amex_metric_mod(y_val.values, oof_preds)
    return accuracy, bool(accuracy > 0.6)


@pipeline(enable_cache=False)
def jz_training_pipeline(
        fetch_train_data,
        fetch_val_data,
        fetch_label_data,
        feature_engineer_train,
        feature_engineer_val,
        training_data_preparation,
        train_xgb_model,
        evaluate_model,
        model_deployer
):
    train_df = fetch_train_data()
    val_df = fetch_val_data()
    label = fetch_label_data()
    train_feat = feature_engineer_train(train_df)
    val_feat = feature_engineer_val(val_df)
    x_train, y_train, x_val, y_val = training_data_preparation(train_feat, val_feat, label)
    model = train_xgb_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    _, deployment_decision = evaluate_model(model, x_val, y_val)
    model_deployer(deployment_decision, model)

if __name__ == "__main__":
    predictor_service_config = PredictionServiceLoaderStepConfig(
        pipeline_name="training_pipeline",
        step_name="model_deployer",
        model_name="xgboost"
    )

    pipe = jz_training_pipeline(
        fetch_train_data=fetch_train_data(config=FetchDataConfig()),
        fetch_val_data=fetch_val_data(),
        fetch_label_data=fetch_label_data(),
        feature_engineer_train=feature_engineer_train(),
        feature_engineer_val=feature_engineer_val(),
        training_data_preparation=training_data_preparation(),
        train_xgb_model=train_xgb_model(),
        evaluate_model=evaluator(),
        model_deployer=mlflow_model_deployer_step()
    )
    pipe.run()