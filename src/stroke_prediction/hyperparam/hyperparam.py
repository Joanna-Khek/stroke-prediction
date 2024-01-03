import pandas as pd
import numpy as np
from loguru import logger

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from hyperopt import STATUS_OK, Trials, hp, tpe, fmin, partial

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


from stroke_prediction.config.settings import config
from stroke_prediction.model_dispatcher import model_selector, param_space
from stroke_prediction.pipeline import pipeline
from stroke_prediction.processing.data_manager import load_dataset
from stroke_prediction.processing.validation import validate_inputs

def hyperparam_search(X: pd.DataFrame, y: pd.DataFrame, model_name: str, param_space: dict):
    """Conduct experimental tracking to find best parameters using MLflow"""
    
    def objective(params):
        with mlflow.start_run():
            
            mlflow.log_params(params)
            
            k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.model.random_state)
            val_fold_score = []
            train_fold_score = []
            
            for fold, (train_idx, val_idx) in enumerate(k_fold.split(X, y)):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                
                # Configure pipeline to add model to the last step
                model_pipeline = (model_name, 
                                       model_selector[model_name].set_params(**params))
                full_pipeline = Pipeline(pipeline.steps + [model_pipeline])

                # Fit pipeline to training data
                full_pipeline.fit(X_train, y_train)

                # Predict
                y_val_pred = full_pipeline.predict_proba(X_val)[:, 1]
                y_train_pred = full_pipeline.predict_proba(X_train)[:, 1]

                # Evaluate
                score_val = roc_auc_score(y_val, y_val_pred)
                score_train = roc_auc_score(y_train, y_train_pred)

                val_fold_score.append(score_val)
                train_fold_score.append(score_train)
            
            mlflow.log_metric("avg_training_roc", np.mean(train_fold_score))
            mlflow.log_metric("avg_val_roc", np.mean(val_fold_score))

        return {'loss': np.mean(val_fold_score), 'status': STATUS_OK}
    
        
    rstate = np.random.default_rng(42)

    best_result = fmin(
        fn = objective, # function to optimize
        space = param_space, 
        algo = tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
        max_evals = 50, # maximum number of iterations
        trials = Trials(), # logging
        rstate = rstate # fixing random state for the reproducibility
    )                


def get_best_params(experiment_name: str, 
                    integer_list: list):
    """Locate the run with lowest average validation MSE and get the params"""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["metrics.avg_val_roc DESC"])[0]
    
    best_metrics = best_run.data.metrics
    best_params = best_run.data.params
    
    # convert to correct type
    for k, v in best_params.items():
        if k not in integer_list:
            best_params[k] = float(v)
        else:
            best_params[k] = int(v)
            
    return best_params, best_metrics

if __name__ == "__main__":

    data = load_dataset(file_name=config.app.training_data_file)
    validated_data, errors = validate_inputs(data)

    X_train, X_test, y_train, y_test = train_test_split(
        validated_data[config.model.features],
        validated_data[config.model.target],
        test_size=config.model.test_size,
        random_state=config.model.random_state # train test split involves randomness
    )

    EXPERIMENT_NAME = f'{config.model.selected_model_name}'
    mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment_tag("version", "1.0")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Search for best params
    logger.info("Hyperparam Searching...")
    hyperparam_search(X_train,
                      y_train,
                      model_name=config.model.selected_model_name,
                      param_space=param_space[config.model.selected_model_name])
    
    logger.info("Getting best params...")
    best_params, best_metrics = get_best_params(experiment_name=EXPERIMENT_NAME,
                                                integer_list=config.model.selected_model_integer_list)

    logger.info(f"The best parameters are: {best_params}")
    logger.info(f"Best ROC is: {best_metrics}")
