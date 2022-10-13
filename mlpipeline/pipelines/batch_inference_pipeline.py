from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def inference_pipeline(
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
    ...
