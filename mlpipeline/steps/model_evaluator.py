from zenml.steps import Output, step
import xgboost as xgb
import pandas as pd

from mlpipeline.steps.util import amex_metric_mod


@step
def evaluator(model: xgb.core.Booster, x_val: pd.DataFrame, y_val:pd.Series) -> Output(accuracy=float):
    valid_dmatrix = xgb.DMatrix(data=x_val, label=y_val)

    oof_preds = model.predict(valid_dmatrix)
    return amex_metric_mod(y_val.values, oof_preds)
