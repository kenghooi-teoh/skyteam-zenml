import mlflow
import pandas as pd
import xgboost as xgb
from zenml.steps import Output, step

from mlpipeline.steps.util import amex_metric_mod

SEED = 123


@step(experiment_tracker="mlflow_tracker")
def train_xgb_model(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> Output(model=xgb.core.Booster):
    """
    Args:
    Returns:
        model:
    """
    mlflow.xgboost.autolog()

    xgb_params = {
        'learning_rate': 0.05,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'random_state': SEED
    }

    train_dmatrix = xgb.DMatrix(data=x_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(data=x_val, label=y_val)

    model = xgb.train(xgb_params, dtrain=train_dmatrix,
                      evals=[(train_dmatrix, 'train'), (valid_dmatrix, 'valid')],
                      verbose_eval=100)

    oof_preds = model.predict(valid_dmatrix)
    amex_metric_mod_scores = amex_metric_mod(y_val.values, oof_preds)

    mlflow.log_metrics({"amex_metric": amex_metric_mod_scores})

    return model


