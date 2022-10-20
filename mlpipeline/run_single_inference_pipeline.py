from datetime import datetime

from pipelines.single_inference_pipeline import single_inference_pipeline
from steps.data_fetcher import fetch_single_inference_data, SingleCustomerQueryConfig
from steps.feature_engineer import feature_engineer_inference_single
from steps.prediction_service_loader import prediction_service_loader, PredictionServiceLoaderStepConfig
from steps.prediction_storer import single_prediction_storer
from steps.predictor import predictor


def run_single_inference_pipeline():
    print("running single inference pipeline")
    fetch_inference_data_config = SingleCustomerQueryConfig(
        customer_id="d96e388ddf6a063e3356fe06db0aa7a52c4a13e3cd34803f1dd25fc99b9003d5"
    )

    predictor_service_config = PredictionServiceLoaderStepConfig(
        pipeline_name="training_pipeline",
        step_name="model_deployer",
        model_name="xgboost"
    )

    pipe = single_inference_pipeline(
        inference_data_fetcher=fetch_single_inference_data(config=fetch_inference_data_config),
        feature_engineer=feature_engineer_inference_single(),
        prediction_service_loader=prediction_service_loader(config=predictor_service_config),
        predictor=predictor(),
        prediction_storer=single_prediction_storer()
    )
    pipe.run()


if __name__ == "__main__":
    run_single_inference_pipeline()
