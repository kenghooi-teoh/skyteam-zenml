import os

import streamlit as st
import streamlit.components.v1 as components

from st_utils import add_logo
add_logo()

DIR = os.path.dirname(os.path.abspath(__file__))

st.title(f"Monitoring Dashboard")

HtmlFile = open("data_drift_report.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
components.html(source_code, height=3000, width=1080)

