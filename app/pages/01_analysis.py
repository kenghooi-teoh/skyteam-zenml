import logging
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine
from zenml.client import Client



connection_string = 'mysql+pymysql://root:root@127.0.0.1:3306/zenml'
engine = create_engine(connection_string)

@st.experimental_memo
def feature_engineer(df: pd.DataFrame):
    all_cols = [c for c in list(df.columns) if c not in ['customer_ID', 'S_2']]
    cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
    num_features = [col for col in all_cols if col not in cat_features]
    valid_cat_features = [fea for fea in cat_features if fea in all_cols]

    test_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

    test_cat_agg = pd.DataFrame()
    if len(valid_cat_features) != 0:
        test_cat_agg = df.groupby("customer_ID")[valid_cat_features].agg(['count', 'last', 'nunique'])
        test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]

    df = pd.concat([test_num_agg, test_cat_agg], axis=1)
    del test_num_agg, test_cat_agg
    return df


def to_date_string(dt: datetime):
    return dt.strftime("%Y-%m-%d")

@st.experimental_memo
def get_default():
    sql = f"""
    SELECT A.customer_ID, A.max_s2, (A.max_s2 + interval 2 month) as default_date, B.target
        FROM 
            (SELECT customer_ID, MAX(S_2) as max_s2 
                FROM customers 
                GROUP BY customer_ID 
            )
        AS A 
    INNER JOIN (SELECT customer_ID, target FROM labels) AS B
    ON A.customer_ID=B.customer_ID
    """

    df = pd.read_sql(sql, engine)
    return df

@st.experimental_memo
def get_default_by_date(start_date: datetime, end_date: datetime):
    sql = f"""
    SELECT A.customer_ID, A.max_s2, (A.max_s2 + interval 2 month) as default_date, B.target
        FROM 
            (SELECT customer_ID, MAX(S_2) as max_s2 
                FROM customers 
                GROUP BY customer_ID 
            )
        AS A 
    INNER JOIN (SELECT customer_ID, target FROM labels) AS B
    ON A.customer_ID=B.customer_ID
    HAVING default_date >= "{to_date_string(start_date)}" AND default_date <= "{to_date_string(end_date)}"
    """

    df = pd.read_sql(sql, engine)
    df = df.sort_values('default_date', ascending=True)
    return df

def get_customers_by_date_range(start_date, end_date):
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

@st.experimental_memo
def get_latest_default_rate_change(df):
    """latest default rate change compare to last month"""
    tmp = df.copy()
    tmp['year'] = tmp['default_date'].dt.year
    tmp['month'] = tmp['default_date'].dt.month
    default_rate_df = tmp.groupby(['year', 'month', 'target'])['customer_ID'].count() / tmp.groupby(['year', 'month'])[
        'customer_ID'].count()
    default_rate_df = default_rate_df.reset_index()
    default_rate_df = default_rate_df.query("target==1").reset_index(drop=True)
    default_rate_df = default_rate_df.rename({'customer_ID': 'default_rate'}, axis=1)
    default_rate_df['prev_default_rate'] = default_rate_df['default_rate'].shift(1)
    latest_rate_change = (default_rate_df.iloc[-1]['default_rate'] - default_rate_df.iloc[-1]['prev_default_rate']) / \
                         default_rate_df.iloc[-1]['prev_default_rate'] * 100
    return latest_rate_change

@st.experimental_memo
def get_latest_default_rate_by_month(df):
    """latest default rate"""
    tmp = df.copy()
    tmp['year'] = tmp['default_date'].dt.year
    tmp['month'] = tmp['default_date'].dt.month
    default_rate_df = tmp.groupby(['year', 'month', 'target'])['customer_ID'].count() / tmp.groupby(['year', 'month'])[
        'customer_ID'].count()
    default_rate_df = default_rate_df.reset_index()
    default_rate_df = default_rate_df.query("target==1").reset_index(drop=True)
    default_rate_df = default_rate_df.rename({'customer_ID': 'default_rate'}, axis=1)
    latest_default_rate = default_rate_df.iloc[-1]['default_rate']
    return latest_default_rate

@st.experimental_memo
def get_latest_default_rate_by_year(df):
    """latest default rate"""
    tmp = df.copy()
    tmp['year'] = tmp['default_date'].dt.year
    tmp['month'] = tmp['default_date'].dt.month
    default_rate_df = tmp.groupby(['year', 'target'])['customer_ID'].count() / tmp.groupby(['year'])[
        'customer_ID'].count()
    default_rate_df = default_rate_df.reset_index()
    default_rate_df = default_rate_df.query("target==1").reset_index(drop=True)
    default_rate_df = default_rate_df.rename({'customer_ID': 'default_rate'}, axis=1)
    latest_default_rate = default_rate_df.iloc[-1]['default_rate']
    return latest_default_rate

@st.experimental_memo
def get_default_rate_by_month(df):
    """latest default rate"""
    tmp = df.copy()
    tmp['year'] = tmp['default_date'].dt.year
    tmp['month'] = tmp['default_date'].dt.month
    default_rate_df = tmp.groupby(['year', 'month', 'target'])['customer_ID'].count() / tmp.groupby(['year', 'month'])[
        'customer_ID'].count()
    default_rate_df = default_rate_df.reset_index()
    default_rate_df = default_rate_df.query("target==1").reset_index(drop=True)
    default_rate_df = default_rate_df.rename({'customer_ID': 'default_rate'}, axis=1)
    default_rate_df['year_month'] = pd.to_datetime(default_rate_df[['year', 'month']].assign(DAY=1))
    default_rate_df = default_rate_df.drop(['year', 'month'], axis=1)
    default_rate_df = default_rate_df[['year_month', 'default_rate']]
    return default_rate_df


def raw_pred_to_class(pred: Union[np.array, list]):
    """
    list of raw pred to int class
    :param pred: list or np.array
    :return: list
    """
    return list(map(lambda x: int(x >= 0.5), pred))


def predict_future(df, horizon_days=31):
    current_last_date = max(df['default_date'])
    prediction_start_date = current_last_date + relativedelta(months=1)
    prediction_start_date = prediction_start_date.replace(day=1)  # get the first date of next month
    prediction_end_date = prediction_start_date + relativedelta(day=horizon_days)  # get the last date of the next month
    feature_start_date = prediction_start_date - relativedelta(months=2)
    feature_end_date = prediction_end_date - relativedelta(months=2)
    logging.info(f'current last date: {current_last_date}, predict data from {prediction_start_date} to {prediction_end_date}, get data from {feature_start_date} to {feature_end_date}')
    features = get_customers_by_date_range(feature_start_date, feature_end_date)
    data = feature_engineer(features)

    client = Client()
    model_deployer = client.active_stack.model_deployer
    if not model_deployer:
        raise RuntimeError("No Model Deployer was found in the active stack.")

    existing_services = model_deployer.find_model_server()

    if existing_services:
        service = existing_services[0]

    else:
        raise RuntimeError(
            f"No MLflow prediction service deployed"
        )

    request_input = np.array(data.to_dict(orient='records'))

    prediction = service.predict(request_input)
    predicted_class_list = raw_pred_to_class(prediction)

    cust_id_list = data.index.to_list()

    predicted_cust_list = [{"target": cls, "customer_ID": cus} for cls, cus in zip(predicted_class_list, cust_id_list)]

    predicted_cust_array = np.array(predicted_cust_list)
    result_df = pd.DataFrame(list(predicted_cust_array))
    result_df = pd.merge(result_df, features, on=["customer_ID"], how="left")[["S_2", "customer_ID", "target"]]
    result_df = result_df.rename({'S_2': 'default_date'}, axis=1)
    return result_df

# Defining the Add Months function
def add_months(start_date, delta_months):
    """
    Add months to date
    """
    end_date = start_date + relativedelta(months=delta_months)
    return end_date

df = get_default_by_date(datetime(2021, 5, 1), datetime(2022, 10, 31))
trend = get_latest_default_rate_change(df)
default_rate_this_month = get_latest_default_rate_by_month(df)
default_rate_this_year = get_latest_default_rate_by_year(df)

# ----- Display -----
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Latest trend against previous month", f"{round(trend, 2)} %")
with col2:
    st.metric("Default rate this month", round(default_rate_this_month, 2))
with col3:
    st.metric("Default rate this year", round(default_rate_this_year, 2))

historic_default_rate_by_m = get_default_rate_by_month(df)
historic_default_rate_by_m['type'] = 'historic'
future_df = predict_future(df)
future_df = future_df.groupby('customer_ID').max().reset_index()
future_default_rate_by_m = get_default_rate_by_month(future_df)
future_default_rate_by_m['year_month'] = future_default_rate_by_m.apply(
    lambda row: add_months(row['year_month'], delta_months=2), axis=1)
future_default_rate_by_m['type'] = 'prediction'
plot_df = pd.concat([historic_default_rate_by_m, future_default_rate_by_m], axis=0)

st.markdown('---')
st.subheader('Default Rate by Years and Months')
fig = px.bar(x='year_month', y='default_rate', data_frame=plot_df, color='type')
st.plotly_chart(fig)

st.markdown('---')
st.subheader('Possible Default Customer IDs')
st.dataframe(future_df['customer_ID'])