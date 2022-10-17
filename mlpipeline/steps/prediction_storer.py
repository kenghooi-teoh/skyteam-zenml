from datetime import datetime

import numpy as np

import pandas as pd
from mlpipeline.steps.util import run_id_to_datetime, save_prediction_in_db


from zenml.environment import Environment
from zenml.steps import (
    Output, step,
    STEP_ENVIRONMENT_NAME,
    StepEnvironment,
BaseParameters
)
from typing import cast


class DataDateFilterConfig(BaseParameters):
    start_date: str
    end_date: str


@step
def prediction_storer(
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
    pipeline_name = step_env.pipeline_name
    run_id = step_env.pipeline_run_id
    step_name = step_env.step_name

    # print("pipeline_name", pipeline_name)
    # print("run_id", run_id)
    # print("step_name", step_name)

    print(data_date_filter_config.start_date, " - ", data_date_filter_config.end_date)
    print(predicted_cust_array)
    print(run_id)
    inference_date = run_id_to_datetime(run_id)
    print(inference_date)

    df = pd.DataFrame(predicted_cust_array)
    df["run_id"] = run_id
    df["inference_date"] = inference_date
    df["data_start_date"] = data_date_filter_config.start_date
    df["data_end_date"] = data_date_filter_config.end_date

    print(f"final output to write to DB: \n {df.shape}")
    print(df.head())

    save_prediction_in_db(df)

    return df





