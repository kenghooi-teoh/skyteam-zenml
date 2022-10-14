from zenml.client import Client
from zenml.services import BaseService
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.steps import step, BaseParameters
from zenml.services.utils import load_last_service_from_step


class PredictionServiceLoaderStepConfig(BaseParameters):
    """Model deployment service loader configuration
    Attributes:
        pipeline_name: name of the pipeline that deployed the model prediction
            server
        step_name: the name of the step that deployed the model prediction
            server
        model_name: the name of the model that was deployed
    """

    pipeline_name: str
    step_name: str
    model_name: str


@step(enable_cache=False)
def prediction_service_loader(
    config: PredictionServiceLoaderStepConfig,
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline"""

    client = Client()
    model_deployer = client.active_stack.model_deployer
    print(model_deployer)
    if not model_deployer:
        raise RuntimeError("No Model Deployer was found in the active stack.")

    service = load_last_service_from_step(
        pipeline_name=config.pipeline_name,
        step_name=config.step_name,
        running=False,
    )

    if not service:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{config.step_name} step in the {config.pipeline_name} pipeline "
            f"is currently running."
        )

    return service


# @step(enable_cache=False)
# def prediction_service_loader(
#         config: PredictionServiceLoaderStepConfig,
# ) -> MLFlowDeploymentService:
#         """Get the prediction service started by the deployment pipeline"""
#
#         client = Client()
#         model_deployer = client.active_stack.model_deployer
#         print(model_deployer)
#         if not model_deployer:
#             raise RuntimeError("No Model Deployer was found in the active stack.")
    # services = model_deployer.find_model_server(
    #     pipeline_name=config.pipeline_name,
    #     pipeline_step_name=config.step_name,
    #     model_name=config.model_name,
    # )
    #
    # if not services:
    #     raise RuntimeError(
    #         f"No model prediction server deployed by the "
    #         f"'{config.step_name}' step in the '{config.pipeline_name}' "
    #         f"pipeline for the '{config.model_name}' model is currently "
    #         f"running."
    #     )
    #
    # if not services[0].is_running:
    #     raise RuntimeError(
    #         f"The model prediction server last deployed by the "
    #         f"'{config.step_name}' step in the '{config.pipeline_name}' "
    #         f"pipeline for the '{config.model_name}' model is not currently "
    #         f"running."
    #     )
    #
    # return services[0]
