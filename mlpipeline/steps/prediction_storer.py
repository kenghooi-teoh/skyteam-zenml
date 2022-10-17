from datetime import datetime

import numpy as np
import pandas as pd
from zenml.steps import Output, step


class StorePredictionConfig:
    data_start_date: datetime
    data_end_date: datetime
    inference_date: datetime
    customer_id: str
    predicted_class: int
    run_id: str
    model_name: str


@step
def prediction_storer(
        prediction,
        config: StorePredictionConfig = None
) -> Output():
    # inference date (datetime obj)
    # metadata e.g. run id, model name
    # predicted class
    # customer id
    # input start date, input end date
    ...

