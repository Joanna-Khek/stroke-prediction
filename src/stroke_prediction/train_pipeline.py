from loguru import logger

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from stroke_prediction.config.settings import config, best_param_config
from stroke_prediction.model_dispatcher import model_selector
from stroke_prediction.pipeline import pipeline
from stroke_prediction.processing.data_manager import load_dataset, save_pipeline
from stroke_prediction.processing.validation import validate_inputs

def run_training(model_name: str) -> None:
    """Train the model using best params from hyperparam tuning

    Args:
        model_name (str): The chosen model to train data and perform prediction
    """

    # Read Training Data
    data = load_dataset(file_name = config.app.training_data_file)

    # Validate Data
    validated_data, errors = validate_inputs(data)

    # Split into Train and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        validated_data[config.model.features],
        validated_data[config.model.target],
        stratify=validated_data[config.model.target],
        test_size=config.model.test_size,
        random_state=config.model.random_state)
    
    # Get best params for chosen model
    best_params = best_param_config[config.model.selected_model_name]

    # Configure pipeline to add model to the last step
    model_pipeline = (model_name, model_selector[model_name].set_params(**best_params))
    full_pipeline = Pipeline(pipeline.steps + [model_pipeline])

    # Fit training data to model pipeline
    full_pipeline.fit(X_train, y_train)

    # Save pipeline
    logger.info("Saving pipeline...")
    save_pipeline(pipeline_to_persist = full_pipeline)

if __name__ == "__main__":
    run_training(model_name = config.model.selected_model_name)

