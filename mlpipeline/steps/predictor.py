# TODO: a predictor class to perform following
# - load model
# - .predict()
# - post processing, model explanation if needed
# - save prediction results for model monitoring (using model_evaluator)

# - write new input to DB
from zenml.steps import Output, step


class Predictor:

    @step
    def load_model(self):
        ...

    def predict(self):
        ...

    def save_prediction(self):
        ...
