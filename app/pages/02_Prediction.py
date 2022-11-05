import streamlit as st

from mlpipeline.pipelines.batch_inference_pipeline import batch_inference_pipeline
from mlpipeline.steps.data_fetcher import fetch_batch_inference_data, FetchDataConfig
from mlpipeline.steps.feature_engineer import feature_engineer
from mlpipeline.steps.prediction_service_loader import prediction_service_loader, PredictionServiceLoaderStepConfig
from mlpipeline.steps.prediction_storer import batch_prediction_storer, DataDateFilterConfig
from mlpipeline.steps.predictor import predictor

st.markdown('Run batch inference')
with st.form("dates"):
    data_start_date = st.date_input("Select start date: ")
    data_end_date = st.date_input("Select end date: ")

    submitted = st.form_submit_button(
        label="Run"
    )

if submitted:
    fetch_inference_data_config = FetchDataConfig(
        start_date=str(data_start_date),
        end_date=str(data_end_date)
    )

    data_date_filter_config = DataDateFilterConfig(
        start_date=str(data_start_date),
        end_date=str(data_end_date)
    )

    predictor_service_config = PredictionServiceLoaderStepConfig(
        pipeline_name="batch_inference_pipeline",
        step_name="model_deployer",
        model_name="xgboost"
    )

    pipe = batch_inference_pipeline(
        inference_data_fetcher=fetch_batch_inference_data(config=fetch_inference_data_config),
        feature_engineer=feature_engineer(),
        prediction_service_loader=prediction_service_loader(config=predictor_service_config),
        predictor=predictor(),
        prediction_storer=batch_prediction_storer(data_date_filter_config=data_date_filter_config)
    )
    pipe.run()

    st.success("Pipeline completed successfully!")



