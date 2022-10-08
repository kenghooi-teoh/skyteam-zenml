import logging

import pandas as pd
from zenml.steps import step


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
