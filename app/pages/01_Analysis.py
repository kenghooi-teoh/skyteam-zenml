import logging
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine

from mlpipeline.pipelines.batch_inference_pipeline import batch_inference_pipeline
from mlpipeline.steps.data_fetcher import fetch_batch_inference_data, FetchDataConfig
from mlpipeline.steps.feature_engineer import feature_engineer

from mlpipeline.steps.prediction_service_loader import PredictionServiceLoaderStepConfig, prediction_service_loader
from mlpipeline.steps.prediction_storer import DataDateFilterConfig, batch_prediction_storer
from mlpipeline.steps.predictor import predictor

connection_string = 'mysql+pymysql://root:root@127.0.0.1:3306/zenml'
engine = create_engine(connection_string)

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

def is_prediction_data_available(prediction_date_start, prediction_date_end):
    prediction_date_start = to_date_string(prediction_date_start)
    prediction_date_end = to_date_string(prediction_date_end)
    sql=f"""
    SELECT
        bim.run_id,
        bim.data_start_date,
        bim.data_end_date
    FROM
        batch_inference_metadata bim
    WHERE bim.data_start_date='{prediction_date_start}' AND bim.data_end_date='{prediction_date_end}';
    """

    df = pd.read_sql(sql, engine)
    if not df.empty:
        return True
    else:
        return False

def get_prediction_df(prediction_date_start, prediction_date_end):
    prediction_date_start = to_date_string(prediction_date_start)
    prediction_date_end = to_date_string(prediction_date_end)
    sql=f"""
    SELECT bi.cust_id AS customer_ID, bim.data_start_date AS default_date, bi.class AS target FROM
    (SELECT
        bim.run_id,
        bim.data_start_date,
        bim.data_end_date
    FROM
        batch_inference_metadata bim
    WHERE bim.data_start_date='{prediction_date_start}' AND bim.data_end_date='{prediction_date_end}') bim LEFT JOIN batch_inference bi ON bim.run_id = bi.run_id;
    """

    df = pd.read_sql(sql, engine)
    df['default_date'] = pd.to_datetime(df['default_date'])
    return df

def predict_future(df, horizon_days=31):
    current_last_date = max(df['default_date'])
    prediction_start_date = current_last_date + relativedelta(months=1)
    prediction_start_date = prediction_start_date.replace(day=1)  # get the first date of next month
    prediction_end_date = prediction_start_date + relativedelta(day=horizon_days)  # get the last date of the next month
    prediction_start_date = prediction_start_date.date()
    prediction_end_date = prediction_end_date.date()
    logging.info(f'current last date: {current_last_date}, predict data from {prediction_start_date} to {prediction_end_date}')
    #query the data if prediction data is available else perform inference
    if is_prediction_data_available(prediction_start_date, prediction_end_date):
        logging.info("Data found from inference table, querying result")
    else:
        fetch_inference_data_config = FetchDataConfig(
            start_date=str(prediction_start_date),
            end_date=str(prediction_end_date)
        )

        data_date_filter_config = DataDateFilterConfig(
            start_date=str(prediction_start_date),
            end_date=str(prediction_end_date)
        )

        predictor_service_config = PredictionServiceLoaderStepConfig(
            pipeline_name="batch_inference_pipeline",
            step_name="model_deployer",
            model_name="xgboost"
        )

        pipe = batch_inference_pipeline(
            inference_data_fetcher=fetch_batch_inference_data(config=fetch_inference_data_config),
            feature_engineer=feature_engineer(),
            prediction_service_loader=prediction_service_loader(config=predictor_service_config),
            predictor=predictor(),
            prediction_storer=batch_prediction_storer(data_date_filter_config=data_date_filter_config)
        )
        pipe.run()
        logging.info("Pipeline completed successfully!")
    result_df = get_prediction_df(prediction_start_date, prediction_end_date)
    return result_df

# Defining the Add Months function
def add_months(start_date, delta_months):
    """
    Add months to date
    """
    end_date = start_date + relativedelta(months=delta_months)
    return end_date

#
date_today = datetime.now().date()
date_one_year_ago = date_today - relativedelta(years=1)
df = get_default_by_date(date_one_year_ago, date_today)
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
future_default_rate_by_m['type'] = 'prediction'
plot_df = pd.concat([historic_default_rate_by_m, future_default_rate_by_m], axis=0)

st.markdown('---')
st.subheader('Default Rate by Years and Months')
fig = px.bar(x='year_month', y='default_rate', data_frame=plot_df, color='type')
st.plotly_chart(fig)

st.markdown('---')
st.subheader('Possible Default Customer IDs')
st.dataframe(future_df['customer_ID'])