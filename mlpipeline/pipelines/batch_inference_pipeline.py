from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def batch_inference_pipeline(
        inference_data_fetcher,
        feature_engineer,
        predictor,
        # post_processor,
        # prediction_storer
):
    """
    Args:

    Returns:
    """
    data = inference_data_fetcher()
    features = feature_engineer(data)
    prediction = predictor(features)
