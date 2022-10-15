# TODO: a predictor class to perform following
# - load model
# - .predict()
# - post processing, model explanation if needed
# - save prediction results for model monitoring (using model_evaluator)

import numpy as np
import pandas as pd
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.steps import Output, step
import json


@step
def predictor(
    service: MLFlowDeploymentService,
    data: pd.DataFrame,
) -> Output():
    """Run a inference request against a prediction service"""

    service.start(timeout=10)
    data = data.fillna(0)  # TODO check feature engineer code first, remove this once fixed

    # https://github.com/zenml-io/zenfiles/blob/main/customer-satisfaction/pipelines/deployment_pipeline.py#L101
    json_list = json.loads(json.dumps(list(data.T.to_dict().values())))
    request_input = np.array(json_list)

    prediction = service.predict(request_input)
    return prediction
