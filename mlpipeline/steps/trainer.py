from sklearn.base import ClassifierMixin
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.steps import Output, step


class Trainer:
    ...

    @enable_mlflow
    @step
    def train_model(self) -> Output(model=ClassifierMixin):
        """
        Args:
        Returns:
            model: ClassifierMixin
        """
        ...
