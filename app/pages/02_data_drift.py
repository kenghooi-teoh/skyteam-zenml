import os, time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from drift_utils import (get_df, last_day)
from datetime import (datetime, date)
from sqlalchemy import create_engine
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

import sys
sys.path.insert(0, '..')
from mlpipeline.steps.util import _feature_engineer # preprocess data
from st_utils import add_logo
add_logo()


engine = create_engine('mysql+pymysql://root:root@127.0.0.1:3306/zenml', echo=False)

st.title(f"Data Drift Monitoring Dashboard")

dt_curr = st.date_input(
    "Please select CURRENT data month",
    date(2022, 10, 1))
st.write('CURRENT data month:', dt_curr)

dt_ref = st.date_input(
    "Please select REFERENCE data month",
    date(2021, 3, 1))
st.write('REFERENCE data month:', dt_ref)


start_date_curr =  str(dt_curr)
end_date_curr = str(last_day(dt_curr))

start_date_ref =  str(dt_ref)
end_date_ref = str(last_day(dt_ref))

st.write('CURRENT data month end:', end_date_curr)
st.write('REFERENCE data month end:', end_date_ref)


src_table = "customers"
df_curr = get_df(start_date_curr, end_date_curr, src_table, engine)
df_ref = get_df(start_date_ref, end_date_ref, src_table, engine)

assert df_curr.shape[0] > 0 
assert df_ref.shape[0] > 0 

df_curr = _feature_engineer(df_curr)
df_ref = _feature_engineer(df_ref)

# drift calculation
my_data_drift_dashboard = Dashboard(tabs=[DataDriftTab(verbose_level=0)])
my_data_drift_dashboard.calculate(df_curr.iloc[:,1:], df_ref.iloc[:,1:], column_mapping=None)
my_data_drift_dashboard.save('data_drift_report.html')

HtmlFile = open("data_drift_report.html", 'r', encoding='utf-8')

source_code = HtmlFile.read() 
components.html(source_code,height=3000,width=1080)




