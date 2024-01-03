from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

from hyperopt import hp
from hyperopt.pyll import scope

from stroke_prediction.config.settings import config

model_selector = {
    'logistic_regression': LogisticRegression(),
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier(),
    'gradient_boosting': GradientBoostingClassifier(),
    'xgboost': xgb.XGBClassifier()  
}

param_space = {
    'gradient_boosting': {
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
        'n_estimators': hp.uniformint('n_estimators', 100, 500),
        'min_samples_split': hp.uniformint('min_samples_split', 2, 10),
        'random_state': config.model.random_state
    }
}