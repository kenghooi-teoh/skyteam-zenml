import logging
from datetime import datetime

import pandas as pd
from zenml.steps import step

from .util import to_date_string


# TODO: change to ingestion from mysql
# 1. DB connector
# 2. query with filters:
#       - date
#       - customer id

class DataFetcher:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass


    @step
    def fetch_train_data(self) -> pd.DataFrame:
        train_feat_df = pd.read_parquet("./src/data/train_importance_fea.parquet")
        return train_feat_df

    @step
    def fetch_val_data(self):
        val_feat_df = pd.read_parquet('./src/data/valid_importance_fea.parquet')
        return val_feat_df

    @step
    def fetch_label_data(self):
        label_df = pd.read_csv('./src/data/train_labels.csv')
        return label_df


def get_customers_by_date_range(start_date, end_date, engine):
    if isinstance(start_date, datetime):
        start_date = to_date_string(start_date)
    if isinstance(end_date, datetime):
        end_date = to_date_string(end_date)
    with engine.begin() as connection:
        data = pd.read_sql(f'select * from customers c where c.new_date >= "{start_date}" and c.new_date <= "{end_date}"', connection)
        return data