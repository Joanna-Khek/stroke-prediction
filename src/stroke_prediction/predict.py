import pandas as pd

from stroke_prediction import __version__ as _version
from stroke_prediction.config.settings import config
from stroke_prediction.processing.data_manager import load_pipeline
from stroke_prediction.processing.validation import validate_inputs

pipeline_file_name = f"{config.app.pipeline_save_file}{_version}.pkl"
_pipe = load_pipeline(file_name = pipeline_file_name)

def make_predictions(input_data: pd.DataFrame) -> dict:
    """Make a prediction using a saved model pipeline"""

    validated_data, errors = validate_inputs(input_data)

    results = {'predictions': None,
               'version': _version,
               'errors': errors}
    
    if not errors:
        predictions = list(_pipe.predict(validated_data[config.model.features]))
        results = {
            'predictions': predictions,
            'version': _version,
            'errors': errors
        }

    return results
