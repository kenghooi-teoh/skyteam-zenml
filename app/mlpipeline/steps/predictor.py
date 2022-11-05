import numpy as np
import pandas as pd
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.steps import Output, step

from mlpipeline.steps.util import raw_pred_to_class


@step
def predictor(
    service: MLFlowDeploymentService,
    data: pd.DataFrame
) -> Output(predicted_cust_array=np.ndarray):
    """Run a inference request against a prediction service"""

    service.start(timeout=10)

    request_input = np.array(data.to_dict(orient='records'))

    prediction = service.predict(request_input)
    predicted_class_list = raw_pred_to_class(prediction)

    cust_id_list = data.index.to_list()

    predicted_cust_list = [{"class": cls, "cust_id": cus} for cls, cus in zip(predicted_class_list, cust_id_list)]

    predicted_cust_array = np.array(predicted_cust_list)
    return predicted_cust_array


