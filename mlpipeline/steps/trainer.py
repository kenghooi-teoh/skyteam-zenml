from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.steps import Output, step
import xgboost as xgb

SEED = 123


class Trainer:
    def __int__(self, x_train, x_val):
        self.x_train = x_train
        self.x_val = x_val

    @enable_mlflow
    @step
    def train_xgb_model(self) -> Output():
        """
        Args:
        Returns:
            model: ClassifierMixin
        """

        xgb_params = {
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'eval_metric': 'logloss',
            'objective': 'binary:logistic',
            'random_state': SEED
        }

        model = xgb.train(xgb_params,
                          dtrain=self.x_train,
                          evals=[(self.x_train, 'train'), (self.x_val, 'valid')],
                          num_boost_round=9999,
                          early_stopping_rounds=100,
                          verbose_eval=100)
        return model