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
    data: pd.DataFrame
) -> Output(predicted_cust_array=np.ndarray):
    """Run a inference request against a prediction service"""

    service.start(timeout=10)
    data = data.fillna(0)  # TODO check feature engineer code first, remove this once fixed
    print("index:\n", data.head().index)

    request_input = np.array(data.to_dict(orient='records'))

    prediction = service.predict(request_input)
    print("raw prediction: ", prediction[:10])
    predicted_class_list = raw_pred_to_class(prediction)

    print("predicted_class: ", list(predicted_class_list)[:10])

    cust_id_list = data.index.to_list()

    predicted_cust_list = [{"class": cls, "cust_id": cus} for cls, cus in zip(predicted_class_list, cust_id_list)]
    print("final output: ", predicted_cust_list[:10])

    predicted_cust_array = np.array(predicted_cust_list)
    return predicted_cust_array
    # return np.array(predicted_class)  # TODO <- [{class:1, inference_date:xxx, cust_id: xxx}]


