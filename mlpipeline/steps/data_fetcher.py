import logging
from datetime import datetime

import pandas as pd
from zenml.steps import step

from .util import to_date_string


# TODO: change to ingestion from postrgres
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

    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv("./data/olist_customers_dataset.csv")
        return df


@step
def ingest_data() -> pd.DataFrame:
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = DataFetcher()
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e


def get_customers_by_date_range(start_date, end_date, engine):
    if isinstance(start_date, datetime):
        start_date = to_date_string(start_date)
    if isinstance(end_date, datetime):
        end_date = to_date_string(end_date)
    with engine.begin() as connection:
        data = pd.read_sql(f'select * from customers c where c.new_date >= "{start_date}" and c.new_date <= "{end_date}"', connection)
        return data