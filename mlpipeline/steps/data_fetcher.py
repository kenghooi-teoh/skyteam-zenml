from datetime import datetime
import pandas as pd
from zenml.steps import Output, step, BaseParameters

from pathlib import Path

from mlpipeline.steps.util import to_date_string, ENGINE

BASE_DIR = Path(__file__).parent.parent.parent.absolute()


class FetchDataConfig(BaseParameters):
    start_date: str = None
    end_date: str = None


class SingleCustomerQueryConfig(BaseParameters):
    customer_id: str


# TODO:
# 1. add customer query by customer id (for single inference)
# 2. logics to fetch batch inference data
@step
def fetch_ondemand_inference_data(config: SingleCustomerQueryConfig) -> Output(data=pd.DataFrame):
    return get_customer_data_by_id(ENGINE, config.customer_id)


@step
def fetch_batch_inference_data(config: FetchDataConfig) -> Output(data=pd.DataFrame):
    print("Getting batch inference data...")
    if config.start_date is None or config.end_date is None:
        data = get_val_data(ENGINE)  # TODO: using get_val_data for now
    else:
        data = get_customers_by_date_range(config.start_date, config.end_date, ENGINE)
    print("Inference data loaded: ", data.shape)
    metadata = {"filter_start_date": config.start_date, "filter_end_date": config.end_date}
    print("inference metadata: ", metadata)

    return data


@step
def print_dataframe_info(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    print(train_df.shape, val_df.shape)

@step
def fetch_train_data(config: FetchDataConfig) -> Output(train_feat_df=pd.DataFrame):
    if config.start_date is None or config.end_date is None:
        train_df = get_training_data(ENGINE)
    else:
        train_df = get_training_data(ENGINE)
        train_df_reduced = select_training_data_by_ratio(train_df)
        new_train_df = get_customers_by_date_range(config.start_date, config.end_date, ENGINE)

        train_df = pd.concat([train_df_reduced, new_train_df], ignore_index=True)
    return train_df

@step
def fetch_val_data(config: FetchDataConfig) -> Output(val_feat_df=pd.DataFrame):
    if config.start_date is None or config.end_date is None:
        val_df = get_val_data(ENGINE)
    else:
        val_df = get_customers_by_date_range(config.start_date, config.end_date, ENGINE)
    return val_df

@step
def fetch_label_data() -> Output(label_df=pd.DataFrame):
    label_df = get_labels(ENGINE)
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


def get_customer_data_by_id(engine, id):
    with engine.begin() as connection:
        data = pd.read_sql('select * from customers where id=id', con=connection)
        return data


def get_labels(engine):
    with engine.begin() as connection:
        data = pd.read_sql('select * from labels', con=connection)
        return data
