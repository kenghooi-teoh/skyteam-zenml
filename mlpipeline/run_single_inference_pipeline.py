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
        customer_id="b2ebc1269b5f7bd62b8ce4f9da74ea9b3eedbedf0aab4447e5a7a00bb127e0eb",
        current_date="2022-12-30"
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
