from datetime import datetime
import pandas as pd

def to_date_string(datetime):
    return datetime.strftime("%Y-%m-%d")

def get_customers_by_date_range(start_date, end_date, engine):
    if isinstance(start_date, datetime):
        start_date = to_date_string(start_date)
    if isinstance(end_date, datetime):
        end_date = to_date_string(end_date)
    with engine.begin() as connection:
        data = pd.read_sql(f'select * from customers c where c.new_date >= "{start_date}" and c.new_date <= "{end_date}"', connection)
        return data