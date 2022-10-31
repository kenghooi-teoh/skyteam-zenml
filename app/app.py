import os, time
import app as st
import pandas as pd
from utils import add_logo

import app.components.v1 as components


add_logo()
DIR = os.path.dirname(os.path.abspath(__file__))

st.title(f"Monitoring Dashboard")

HtmlFile = open("dashboard.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
print(source_code)
components.html(source_code,height=3000,width=1080)

