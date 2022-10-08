import numpy as np
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.steps import Output, step


class ModelEvaluator:
    def __init__(self, model, x_val, y_val):
        self.model = model
        self.x_val = x_val
        self.y_val = y_val

    @enable_mlflow
    @step
    def evaluation(
        self
    ) -> Output:
        """
        Args:
        Returns:
        """
        oof_preds = self.model.predict(self.x_val)
        acc = self.amex_metric_mod(self.y_val.values, oof_preds)
        return acc

    @staticmethod
    def amex_metric_mod(y_true, y_pred):
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, 1].argsort()[::-1]]
        weights = np.where(labels[:, 0] == 0, 20, 1)
        cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
        top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])

        gini = [0, 0]
        for i in [1, 0]:
            labels = np.transpose(np.array([y_true, y_pred]))
            labels = labels[labels[:, i].argsort()[::-1]]
            weight = np.where(labels[:, 0] == 0, 20, 1)
            weight_random = np.cumsum(weight / np.sum(weight))
            total_pos = np.sum(labels[:, 0] * weight)
            cum_pos_found = np.cumsum(labels[:, 0] * weight)
            lorentz = cum_pos_found / total_pos
            gini[i] = np.sum((lorentz - weight_random) * weight)

        return 0.5 * (gini[1] / gini[0] + top_four)

    def interpret(self):
        # TODO: return model explanation
        ...
