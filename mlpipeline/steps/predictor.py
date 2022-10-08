# TODO: a predictor class to perform following
# - load model
# - .predict()
# - post processing, model explanation if needed
# - save prediction results for model monitoring (using model_evaluator)

# - write new input to DB
from zenml.steps import Output, step
import mlflow.pyfunc


class Predictor:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model_name = "xgboost"
        model_version = 1

        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )

        return model

    @step
    def predict(self, x_in):
        return self.model.predict(x_in)

    def save_prediction(self):
        ...
