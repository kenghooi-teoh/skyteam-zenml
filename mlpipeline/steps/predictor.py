# TODO: a predictor class to perform following
# - load model
# - .predict()
# - post processing, model explanation if needed
# - save prediction results for model monitoring (using model_evaluator)
from xgboost import DMatrix, Booster


# - write new input to DB
from zenml.steps import Output, step, BaseParameters


class PredictorConfig(BaseParameters):
    model: Booster
    input_features: DMatrix
    

@step
def predictor(config: PredictorConfig):
    return config.model.predict(config.input_features)
