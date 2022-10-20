from typing import cast

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from zenml.environment import Environment
from zenml.steps import (
    Output, step,
    STEP_ENVIRONMENT_NAME,
    StepEnvironment,
    BaseParameters
)

from mlpipeline.steps.util import run_id_to_datetime, load_df_to_sql

engine = create_engine('mysql+pymysql://root:root@127.0.0.1:3306/zenml', echo=False)


class DataDateFilterConfig(BaseParameters):
    start_date: str
    end_date: str


@step
def single_prediction_storer(
        predicted_cust_array: np.ndarray
) -> Output():

    step_env = cast(StepEnvironment, Environment()[STEP_ENVIRONMENT_NAME])
    run_id = step_env.pipeline_run_id

    inference_date = run_id_to_datetime(run_id.replace("single_inference_pipeline-", ""))

    df = pd.DataFrame(list(predicted_cust_array))
    df["run_id"] = run_id
    df["inference_date"] = inference_date

    with engine.begin() as connection:
        print("saving single prediction and metadata")
        load_df_to_sql(df[["class", "cust_id", "run_id"]], 'single_inference', connection, "append")
        load_df_to_sql(df.loc[[0], ["run_id", "inference_date"]],
                       'single_inference_metadata',
                       connection,
                       "append")

    return df


@step
def batch_prediction_storer(
        data_date_filter_config: DataDateFilterConfig,
        predicted_cust_array: np.ndarray
) -> Output():
    """
    Write to DB
        # [x] inference date (datetime obj)
        # [x] metadata run_id
        # [] metadata model_name
        # [x] predicted class
        # [x] customer id
        # [x] input start date, input end date
    """
    step_env = cast(StepEnvironment, Environment()[STEP_ENVIRONMENT_NAME])
    # pipeline_name = step_env.pipeline_name
    # step_name = step_env.step_name
    run_id = step_env.pipeline_run_id

    print(data_date_filter_config.start_date, " - ", data_date_filter_config.end_date)
    print(predicted_cust_array)
    print(run_id)
    inference_date = run_id_to_datetime(run_id.replace("batch_inference_pipeline-", ""))
    print(inference_date)

    df = pd.DataFrame(list(predicted_cust_array))
    df["run_id"] = run_id
    df["inference_date"] = inference_date
    df["data_start_date"] = data_date_filter_config.start_date
    df["data_end_date"] = data_date_filter_config.end_date

    with engine.begin() as connection:
        load_df_to_sql(df[["class", "cust_id", "run_id"]], 'batch_inference', connection, "append")
        load_df_to_sql(df.loc[[0], ["run_id", "inference_date", "data_start_date", "data_end_date"]],
                       'batch_inference_metadata',
                       connection,
                       "append")

    return df
