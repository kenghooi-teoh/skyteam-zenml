from datetime import datetime

import pandas as pd
from zenml.steps import Output, step, BaseParameters

from pathlib import Path

from mlpipeline.steps.util import to_date_string, engine

BASE_DIR = Path(__file__).parent.parent.parent.absolute()

class FetchDataConfig(BaseParameters):
    start_date:str=None
    end_date:str=None

# TODO: change to ingestion from mysql
# 1. DB connector
# 2. query with filters:
#       - date
#       - customer id
@step
def print_dataframe_info(train_df:pd.DataFrame, val_df:pd.DataFrame) -> None:
    print(train_df.shape, val_df.shape)

@step
def fetch_train_data(config: FetchDataConfig) -> Output(train_feat_df=pd.DataFrame):
    if config.start_date is None or config.end_date is None:
        train_df = get_training_data(engine)
    else:
        train_df = get_training_data(engine)
        train_df_reduced = select_training_data_by_ratio(train_df)
        new_train_df = get_customers_by_date_range(config.start_date, config.end_date, engine)

        train_df = pd.concat([train_df_reduced, new_train_df], ignore_index=True)
    return train_df

@step
def fetch_val_data(config: FetchDataConfig) -> Output(val_feat_df=pd.DataFrame):
    if config.start_date is None or config.end_date is None:
        val_df = get_val_data(engine)
    else:
        val_df = get_customers_by_date_range(config.start_date, config.end_date, engine)
    return val_df

@step
def fetch_label_data() -> Output(label_df=pd.DataFrame):
    label_df = get_labels(engine)
    return label_df


def get_customers_by_date_range(start_date, end_date, engine):
    if isinstance(start_date, datetime):
        start_date = to_date_string(start_date)
    if isinstance(end_date, datetime):
        end_date = to_date_string(end_date)
    with engine.begin() as connection:
        query = f'select * from customers c where c.S_2 >= "{start_date}" and c.S_2 <= "{end_date}"'
        data = pd.read_sql(query, connection)
        return data


def select_training_data_by_ratio(train_df, ratio=0.5):
    train_df_13 = train_df.loc[train_df.groupby('customer_ID')['customer_ID'].transform(len) == 13]

    customer_ids = train_df_13['customer_ID'].unique()
    customer_ids_len = len(customer_ids)
    customer_ids_reduced = customer_ids[:int(customer_ids_len*ratio)]

    return train_df_13[train_df_13['customer_ID'].isin(customer_ids_reduced)]


def get_training_data(engine):
    with engine.begin() as connection:
        data = pd.read_sql('select * from train_data', con=connection)
        return data


def get_val_data(engine):
    with engine.begin() as connection:
        data = pd.read_sql('select * from valid_data', con=connection)
        return data


def get_customers_data(engine):
    with engine.begin() as connection:
        data = pd.read_sql('select * from customers', con=connection)
        return data


def get_labels(engine):
    with engine.begin() as connection:
        data = pd.read_sql('select * from labels', con=connection)
        return data
