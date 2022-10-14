# TODO: a predictor class to perform following
# - load model
# - .predict()
# - post processing, model explanation if needed
# - save prediction results for model monitoring (using model_evaluator)

import pandas as pd
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.steps import Output, step


@step
def predictor(
    service: MLFlowDeploymentService,
    data: pd.DataFrame,
) -> Output():
    """Run a inference request against a prediction service"""

    service.start(timeout=10)
    prediction = service.predict(data.values)  # TODO: make this step predict DMatrix (xbgoost)
    print(f"prediction: {prediction}")
    return prediction
