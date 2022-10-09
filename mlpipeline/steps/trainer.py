from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.steps import Output, step
import mlflow
import xgboost as xgb

import pandas as pd
from .model_evaluator import ModelEvaluator
SEED = 123




@enable_mlflow
@step
def train_xgb_model(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> Output():
    """
    Args:
    Returns:
        model: ClassifierMixin
    """
    mlflow.xgboost.autolog()

    xgb_params = {
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'random_state': SEED
    }

    train_dmatrix = xgb.DMatrix(data=x_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(data=x_val, label=y_val)

    model = xgb.train(xgb_params, dtrain=train_dmatrix,
                      evals=[(train_dmatrix, 'train'),(valid_dmatrix, 'valid')],
                      num_boost_round=9999,
                      early_stopping_rounds=100,
                      verbose_eval=100)

    oof_preds = model.predict(x_val)
    amex_metric_mod_scores = ModelEvaluator.amex_metric_mod(y_val.values, oof_preds)

    # log metrics
    mlflow.log_metrics({"amex_metric": amex_metric_mod_scores})

    mlflow.xgboost.log_model(
        sk_model=model,
        artifact_path="xgboost",
        registered_model_name="xgboost"
    )

    return model


