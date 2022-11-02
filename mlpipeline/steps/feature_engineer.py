import pandas as pd
from zenml.steps import Output, step
from .util import _feature_engineer

@step
def feature_engineer(df: pd.DataFrame) -> Output(train_feat=pd.DataFrame):
    return _feature_engineer(df)

@step
def feature_engineer_train(df: pd.DataFrame) -> Output(train_feat=pd.DataFrame):
    return _feature_engineer(df)

@step
def feature_engineer_val(df: pd.DataFrame) -> Output(train_feat=pd.DataFrame):
    return _feature_engineer(df)
@step
def feature_engineer_inference_single(df: pd.DataFrame) -> Output(train_feat=pd.DataFrame):
    return _feature_engineer(df)
@step
def feature_engineer_inference_batch(df: pd.DataFrame) -> Output(train_feat=pd.DataFrame):
    return _feature_engineer(df)
