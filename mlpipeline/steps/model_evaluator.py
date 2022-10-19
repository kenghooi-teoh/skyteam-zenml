from typing import cast

import numpy as np
import pandas as pd
import xgboost as xgb
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.services import BaseService
from zenml.steps import step, Output, BaseParameters

from mlpipeline.steps.util import amex_metric_mod, raw_pred_to_class


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
def evaluator(model: xgb.core.Booster, service: BaseService, x_val: pd.DataFrame, y_val:pd.Series, is_retraining: bool) -> Output(deployment_decision=bool):
    valid_dmatrix = xgb.DMatrix(data=x_val, label=y_val)

    oof_preds = model.predict(valid_dmatrix)
    accuracy = amex_metric_mod(y_val.values, oof_preds)

    if is_retraining:
        request_input = np.array(x_val.to_dict(orient='records'))
        mlflow_service = cast(MLFlowDeploymentService, service)
        predictions = mlflow_service.predict(request_input)

        predicted_class = raw_pred_to_class(predictions)

        accuracy_current_model = amex_metric_mod(y_val.values, predicted_class)

        print('retraining')
        print('accuracy:', accuracy)
        print('current_model:', accuracy_current_model)

        return bool(accuracy >= accuracy_current_model)

    else:
        print('common training')
        print('accuracy:', accuracy)
        return bool(accuracy > 0.6)