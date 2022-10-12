from datetime import datetime

import pandas as pd
from zenml.steps import Output, step

from sqlalchemy import create_engine
from pathlib import Path

from mlpipeline.steps.util import to_date_string

BASE_DIR = Path(__file__).parent.parent.parent.absolute()


# TODO: change to ingestion from mysql
# 1. DB connector
# 2. query with filters:
#       - date
#       - customer id

ENGINE = create_engine('mysql://root:root@127.0.0.1:3306/zenml', echo=False)

@step
def fetch_train_data() -> Output(train_feat_df=pd.DataFrame):
    with ENGINE.begin() as connection:
        data = pd.read_sql('select * from train_data', con=connection)
        return data

@step
def fetch_val_data() -> Output(val_feat_df=pd.DataFrame):
    with ENGINE.begin() as connection:
        data = pd.read_sql('select * from valid_data', con=connection)
        return data

@step
def fetch_label_data() -> Output(label_df=pd.DataFrame):
    with ENGINE.begin() as connection:
        data = pd.read_sql('select * from labels', con=connection)
        return data


def get_customers_by_date_range(start_date, end_date, engine):
    if isinstance(start_date, datetime):
        start_date = to_date_string(start_date)
    if isinstance(end_date, datetime):
        end_date = to_date_string(end_date)
    with engine.begin() as connection:
        data = pd.read_sql(f'select * from customers c where c.new_date >= "{start_date}" and c.new_date <= "{end_date}"', connection)
        return data


# def get_training_data(engine):
#     with engine.begin() as connection:
#         data = pd.read_sql('select * from train_data', con=connection)
#         return data
#
#
# def get_val_data(engine):
#     with engine.begin() as connection:
#         data = pd.read_sql('select * from valid_data', con=connection)
#         return data
#
#
# def get_labels(engine):
#     with engine.begin() as connection:
#         data = pd.read_sql('select * from labels', con=connection)
#         return data
