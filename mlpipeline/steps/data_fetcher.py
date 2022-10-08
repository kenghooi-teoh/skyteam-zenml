import logging
from datetime import datetime

import pandas as pd
from zenml.steps import Output, step

from steps.util import to_date_string
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.absolute()


# TODO: change to ingestion from mysql
# 1. DB connector
# 2. query with filters:
#       - date
#       - customer id

@step
def fetch_train_data() -> Output(train_feat_df=pd.DataFrame):
    train_feat_df = pd.read_parquet(BASE_DIR.joinpath("data/train_importance_fea.parquet"))
    return train_feat_df

@step
def fetch_val_data() -> Output(val_feat_df=pd.DataFrame):
    val_feat_df = pd.read_parquet(BASE_DIR.joinpath('data/valid_importance_fea.parquet'))
    return val_feat_df

@step
def fetch_label_data() -> Output(label_df=pd.DataFrame):
    label_df = pd.read_csv(BASE_DIR.joinpath("data/train_labels.csv"))
    print("import done")
    return label_df


def get_customers_by_date_range(start_date, end_date, engine):
    if isinstance(start_date, datetime):
        start_date = to_date_string(start_date)
    if isinstance(end_date, datetime):
        end_date = to_date_string(end_date)
    with engine.begin() as connection:
        data = pd.read_sql(f'select * from customers c where c.new_date >= "{start_date}" and c.new_date <= "{end_date}"', connection)
        return data


def get_training_data(engine):
    with engine.begin() as connection:
        data = pd.read_sql('select * from customers', con=connection)
        return data


def get_labels(engine):
    with engine.begin() as connection:
        data = pd.read_sql('select * from labels', con=connection)
        return data
