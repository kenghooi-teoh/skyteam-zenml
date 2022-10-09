from zenml.steps import Output, step
import mlflow
import xgboost as xgb

import pandas as pd
from .model_evaluator import ModelEvaluator
SEED = 123


@step
def train_xgb_model(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> Output():
    """
    Args:
    Returns:
        model: ClassifierMixin
    """
    # autologging does not work because of No module named 'matplotlib'
    # mlflow.xgboost.autolog()

    xgb_params = {
        'learning_rate': 0.05,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'random_state': SEED
    }

    train_dmatrix = xgb.DMatrix(data=x_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(data=x_val, label=y_val)

    model = xgb.train(xgb_params, dtrain=train_dmatrix,
                      evals=[(train_dmatrix, 'train'),(valid_dmatrix, 'valid')],
                      verbose_eval=100)

    oof_preds = model.predict(valid_dmatrix)
    amex_metric_mod_scores = ModelEvaluator.amex_metric_mod(y_val.values, oof_preds)

    # log metrics
    mlflow.log_metrics({"amex_metric": amex_metric_mod_scores})

    # MlflowException: Model Registry features are not supported by the store with URI
    # mlflow.xgboost.log_model(
    #     xgb_model=model,
    #     artifact_path="xgboost",
    #     registered_model_name="xgboost"
    # )

    return model


