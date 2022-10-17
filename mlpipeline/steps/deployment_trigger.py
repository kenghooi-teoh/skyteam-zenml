from zenml.steps import step


@step
def deployment_trigger(val_acc: float) -> bool:
    print("deployment decision: ", val_acc > 0.6)
    return val_acc > 0.6
