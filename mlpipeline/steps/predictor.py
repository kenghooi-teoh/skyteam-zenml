# TODO: a predictor class to perform following
# - load model
# - .predict()
# - post processing, model explanation if needed
# - save prediction results for model monitoring (using model_evaluator)

import numpy as np
import pandas as pd
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.steps import Output, step
from mlpipeline.steps.util import raw_pred_to_class


@step
def predictor(
    service: MLFlowDeploymentService,
    data: pd.DataFrame,
) -> Output():
    """Run a inference request against a prediction service"""

    service.start(timeout=10)
    data = data.fillna(0)  # TODO check feature engineer code first, remove this once fixed

    request_input = np.array(data.to_dict(orient='records'))

    prediction = service.predict(request_input)
    predicted_class = map(raw_pred_to_class, prediction)
    print("predicted_class: ", list(predicted_class))
    return np.array(predicted_class)


