from zenml.steps import step


@step
def deployment_trigger(val_acc: float) -> bool:
    return val_acc > 0.6

@step
def retraining_deployment_trigger(current_acc: float, new_acc: float):
    return current_acc < new_acc