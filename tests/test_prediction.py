import numpy as np
from typing import List

from stroke_prediction.predict import make_predictions

def test_make_prediction(sample_input_data):

    result = make_predictions(input_data = sample_input_data)
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.int64)


