import pandas as pd
from datetime import datetime
from datetime import (datetime, timedelta)

def get_df(start_date:str, end_date:str, src_table:str,engine) -> pd.DataFrame: 
    query = f'''
            select c2.* from 
            (
            select c.customer_ID as customer_ID from {src_table} c 
            group by c.customer_ID 
            having max(c.S_2) between "{start_date}" and "{end_date}"   
            ) c_id 
            left join {src_table} c2 
            on c2.customer_ID = c_id.customer_ID
            '''

    with engine.begin() as connection:
        # print(query)
        data = pd.read_sql(query, con=connection)
        # print(data.shape)
        return data


def last_day(your_date: datetime)-> datetime:
    # print("The original date is : " + str(your_date))
    nxt_mnth = your_date.replace(day=28) + timedelta(days=4)
    # subtracting the days from next month date to
    # get last date of current Month
    return nxt_mnth - timedelta(days=nxt_mnth.day)