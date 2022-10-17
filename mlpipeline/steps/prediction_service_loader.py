from typing import cast

from zenml.client import Client
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.steps import step, BaseParameters


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
    print("model_deployer in active stack: ", model_deployer)
    if not model_deployer:
        raise RuntimeError("No Model Deployer was found in the active stack.")

    existing_services = model_deployer.find_model_server(
    )
    print("existing services: ", existing_services)
    print(model_deployer.find_model_server())

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])

    else:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{config.step_name} step in the {config.pipeline_name} pipeline "
            f"is currently running."
        )

    return service
