import pandas as pd

import numpy as np
from dateutil.relativedelta import relativedelta

import logging
logging.basicConfig(level=logging.INFO, format=logging.BASIC_FORMAT)

from sqlalchemy import create_engine

logger = logging.getLogger("load_data")

engine = create_engine('mysql://root:root@127.0.0.1:3306/zenml', echo=False)

train_df = pd.read_parquet('./data/train_importance_fea.parquet')
valid_df = pd.read_parquet('./data/valid_importance_fea.parquet')
other_df = pd.read_parquet('./data/other_importance_fea.parquet')

train_df['S_2'] = pd.to_datetime(train_df['S_2'])
valid_df['S_2'] = pd.to_datetime(valid_df['S_2'])
other_df['S_2'] = pd.to_datetime(other_df['S_2'])

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
    

other_df = create_dummy_customer_data(other_df)

with engine.begin() as connection:
    logger.info("ingesting training data")
    train_df.to_sql('train_data', con=connection, if_exists='replace', index=False)

    logger.info("ingesting validation data")
    valid_df.to_sql('valid_data', con=connection, if_exists='replace', index=False)

    logger.info('ingesting dummy customer data')
    other_df.to_sql('customers', con=connection, if_exists='replace', index=False)



