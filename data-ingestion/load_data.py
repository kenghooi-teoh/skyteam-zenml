import pandas as pd

import numpy as np
from dateutil.relativedelta import relativedelta

import logging
logging.basicConfig(level=logging.INFO, format=logging.BASIC_FORMAT)

from sqlalchemy import create_engine

logger = logging.getLogger("load_data")

engine = create_engine('mysql+pymysql://root:root@127.0.0.1:3306/zenml', echo=False)

def reduce_months(start_date, delta_period):
    end_date = start_date - relativedelta(months=delta_period)
    return end_date

def create_dummy_customer_data(input_df):
    input_df['add_year'] = input_df['S_2'] + pd.offsets.DateOffset(years=5)
    customer_id_distinct = input_df['customer_ID'].value_counts().index.to_list()

    customer_id_split = np.array_split(customer_id_distinct, 24)
    month_range = range(1, 24+1)

    dfs = []

    for ids, month_to_reduce in zip(customer_id_split, month_range):
        df = input_df[input_df['customer_ID'].isin(ids)].copy()
        df["S_2"] = df.apply(lambda row: reduce_months(row["add_year"], month_to_reduce), axis = 1)
        dfs.append(df)

    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined = df_combined.drop(['add_year'], axis=1)

    return df_combined

def load_parquet_to_sql(parquet_data_path, table_name, connection, pre_processing=None):
    df = pd.read_parquet(parquet_data_path)
    df['S_2'] = pd.to_datetime(df['S_2'])

    if pre_processing:
        df = pre_processing(df)

    df.to_sql(table_name, con=connection, if_exists='replace', index=False)


def load_csv_to_sql(csv_data_path, table_name, con, pre_processing=None):
    df = pd.read_csv(csv_data_path)

    df.to_sql(table_name, con=con, if_exists='replace', index=False)


with engine.begin() as connection:
    logger.info("ingesting training data")
    load_parquet_to_sql('./data/train_importance_fea.parquet', 'train_data', connection)

    logger.info("ingesting validation data")
    load_parquet_to_sql('./data/valid_importance_fea.parquet', 'valid_data', connection)

    logger.info('ingesting dummy customer data')
    load_parquet_to_sql('./data/other_importance_fea.parquet', 'customers', connection, create_dummy_customer_data)

    logger.info('ingesting dummy label data')
    load_csv_to_sql('./data/train_labels.csv', 'labels', connection)