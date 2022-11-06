from datetime import datetime
from pathlib import Path

import pandas as pd
from zenml.steps import Output, step, BaseParameters

from mlpipeline.steps.util import to_date_string, ENGINE

BASE_DIR = Path(__file__).parent.parent.parent.absolute()


class FetchDataConfig(BaseParameters):
    start_date: str = None
    end_date: str = None


class SingleCustomerQueryConfig(BaseParameters):
    customer_id: str
    current_date: str


@step
def fetch_single_inference_data(config: SingleCustomerQueryConfig) -> Output(data=pd.DataFrame):
    cust_payments_by_month = get_customer_data_by_id(ENGINE, config.customer_id, config.current_date)
    if len(cust_payments_by_month) < 6:
        raise Exception("Insufficient historical data for this customer. Must have at least 6 months of data.")
    return cust_payments_by_month


@step
def fetch_batch_inference_data(config: FetchDataConfig) -> Output(data=pd.DataFrame):
    print("Getting batch inference data...")
    if config.start_date is None or config.end_date is None:
        data = get_val_data(ENGINE)
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
        query = f'''
            select c2.* from (
            select c.customer_ID as customer_ID from customers c 
            group by c.customer_ID 
            having max(c.S_2) >= "{start_date}" and max(c.S_2) <= "{end_date}"   
            ) c_id left join customers c2 on c2.customer_ID = c_id.customer_ID
        '''
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


def get_customer_data_by_id(engine, cust_id, current_date):
    with engine.begin() as connection:
        data = pd.read_sql(
            f"select * from customers c where c.customer_ID='{cust_id}' and c.S_2<='{current_date}'",
            con=connection
        )
        return data


def get_labels(engine):
    with engine.begin() as connection:
        data = pd.read_sql('select * from labels', con=connection)
        return data
