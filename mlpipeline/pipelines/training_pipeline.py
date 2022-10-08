from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline

from ..steps.data_fetcher import DataFetcher
from ..steps.dataset_maker import DatasetMaker
from ..steps.trainer import Trainer
from ..steps.feature_engineer import FeatureEngineer
from ..steps.model_evaluator import ModelEvaluator


class TrainingPipelineConfig:
    data_fetcher: DataFetcher
    dataset_maker: DatasetMaker
    trainer: Trainer
    feature_engineer: FeatureEngineer
    evaluator: ModelEvaluator


# TODO
# - eventually write model to model registry with evaluation scores
@pipeline(enable_cache=False, required_integrations=[MLFLOW])
def training_pipeline(config: TrainingPipelineConfig):
    """
    Args:

    Returns:
    """
    ...
