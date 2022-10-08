from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline

from ..steps.data_fetcher import DataFetcher
from ..steps.data_preprocessor import DataPreprocessor
from ..steps.trainer import Trainer
from ..steps.feature_engineer import FeatureEngineer
from ..steps.model_evaluator import ModelEvaluator


class TrainingPipelineConfig:
    data_fetcher: DataFetcher
    trainer: Trainer
    feature_engineer: FeatureEngineer
    evaluator: ModelEvaluator


# TODO
# - eventually write model to model registry with evaluation scores
@pipeline(enable_cache=False, required_integrations=[MLFLOW])
def training_pipeline():
    """
    Args:

    Returns:s
    """
    ...

    train_df = DataFetcher().fetch_train_data()
    val_df = DataFetcher().fetch_val_data()
    label = DataFetcher().fetch_label_data()
    print("data loaded")

    train_feat = FeatureEngineer().feature_engineer(train_df)
    val_feat = FeatureEngineer().feature_engineer(val_df)

    x_train, train_y, x_val, y_val = DataPreprocessor().clean_data(train_feat, val_feat, label)
    print("data preprocessed")

    model = Trainer(x_train=x_train, x_val=x_val, y_val=y_val).train_xgb_model()
    print("model trained")


if __name__ == "main":
    training_pipeline()
