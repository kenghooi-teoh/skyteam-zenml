import numpy as np

from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def batch_inference_pipeline(
        inference_data_fetcher,
        feature_engineer,
        prediction_service_loader,
        predictor
        # post_processor,
        # prediction_storer
):
    """
    Args:

    Returns:
    """
    data = inference_data_fetcher()
    features = feature_engineer(data)
    prediction_service = prediction_service_loader()
    predictor = predictor(service=prediction_service, data=features)
