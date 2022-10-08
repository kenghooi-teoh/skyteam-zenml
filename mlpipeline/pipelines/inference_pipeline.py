from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline

from ..steps.data_fetcher import DataFetcher
from ..steps.predictor import Predictor


class InferencePipelineConfig:
    data_fetcher: DataFetcher
    predictor: Predictor


# TODO
# - eventually write model to model registry with evaluation scores
@pipeline(enable_cache=False, required_integrations=[MLFLOW])
def inference_pipeline(config: InferencePipelineConfig):
    """
    Args:

    Returns:
    """
    ...
