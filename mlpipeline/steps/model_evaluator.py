from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.steps import Output, step


class ModelEvaluator:
    ...

    @enable_mlflow
    @step
    def evaluation(
        self
    ) -> Output:
        """
        Args:
        Returns:
        """
        # TODO: return eval scores
        ...

    def interpret(self):
        # TODO: return model explanation
        ...
