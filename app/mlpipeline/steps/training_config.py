from zenml.steps import step, BaseParameters, Output


class TrainingConfig(BaseParameters):
    is_retraining: bool = False


@step
def training_config(config: TrainingConfig) -> Output(is_retrainig=bool):
    return config.is_retraining
