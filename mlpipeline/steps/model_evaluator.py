import numpy as np

import xgboost as xgb
import pandas as pd

from zenml.client import Client
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.steps import step, StepEnvironment, STEP_ENVIRONMENT_NAME, Output
from zenml.environment import Environment
from typing import cast

from mlpipeline.steps.util import amex_metric_mod


@step
def evaluator(model: xgb.core.Booster, x_val: pd.DataFrame, y_val:pd.Series, is_retraining: bool) -> Output(accuracy=float, deployment_decision=bool):
    if is_retraining:
        # load current model sevice
        step_env = cast(StepEnvironment, Environment()[STEP_ENVIRONMENT_NAME])
        pipeline_name = step_env.pipeline_name
        step_name = step_env.step_name

        client = Client()
        model_deployer = client.active_stack.model_deployer

        model_name = "xgboost"

        current_services = model_deployer.find_model_server(
            pipeline_name=pipeline_name,
            pipeline_step_name=step_name,
            model_name=model_name,
        )
        if current_services:
            service = cast(MLFlowDeploymentService, current_services[0])

        else:
            raise RuntimeError(
                f"No MLflow prediction service deployed by the "
                f"{step_name} step in the {pipeline_name} pipeline "
                f"is currently running."
            )

        request_input = np.array(x_val.to_dict(orient='records'))

        prediction = service.predict(request_input)

        return 0.6, True

    else:
        valid_dmatrix = xgb.DMatrix(data=x_val, label=y_val)

        oof_preds = model.predict(valid_dmatrix)
        accuracy = amex_metric_mod(y_val.values, oof_preds)
        decision = accuracy > 0.6
        decision = bool(decision)

        return accuracy, decision