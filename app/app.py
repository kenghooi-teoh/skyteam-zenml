import os

import streamlit as st
import streamlit.components.v1 as components
from my_utils import get_df, last_day
from datetime import date

from sqlalchemy import create_engine

from st_utils import add_logo
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab


engine = create_engine('mysql+pymysql://root:root@127.0.0.1:3306/zenml', echo=False)

add_logo()
DIR = os.path.dirname(os.path.abspath(__file__))

st.title(f"Data Drift Monitoring Dashboard")

# run_id = mlflow.last_active_run().info.run_id
# dboard_path = os.path.join('mlruns','0',run_id,'artifacts','iris_report.html')
# HtmlFile = open(dboard_path, 'r', encoding='utf-8')

dt_curr = st.date_input(
    "Please select CURRENT data month",
    date(2017, 3, 1))
st.write('CURRENT data month:', dt_curr)

dt_ref = st.date_input(
    "Please select REFERENCE data month",
    date(2018, 3, 1))
st.write('REFERENCE data month:', dt_ref)


src_table = "valid_data"

start_date_curr = str(dt_curr)
end_date_curr = str(last_day(dt_curr))

start_date_ref = str(dt_ref)
end_date_ref = str(last_day(dt_ref))

st.write('CURRENT data month end:', end_date_curr)
st.write('REFERENCE data month end:', end_date_ref)
# print(end_date_curr, end_date_ref)

df_curr = get_df(start_date_curr, end_date_curr, src_table, engine)
df_ref = get_df(start_date_ref, end_date_ref, src_table, engine)

my_data_drift_dashboard = Dashboard(tabs=[DataDriftTab(verbose_level=0)])
my_data_drift_dashboard.calculate(df_curr.iloc[:, 2:], df_ref.iloc[:, 2:], column_mapping=None)
my_data_drift_dashboard.save('sky_report.html')

HtmlFile = open("data_drift_report.html", 'r', encoding='utf-8')
# HtmlFile = open("sky_report.html", 'r', encoding='utf-8')

source_code = HtmlFile.read()
# print(source_code)
components.html(source_code, height=3000, width=1080)

