from zenml.steps import step


@step
def deployment_trigger(val_acc: float) -> bool:
    return val_acc > 0.6
