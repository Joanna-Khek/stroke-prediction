import pytest

from stroke_prediction.config.settings import config
from stroke_prediction.processing.data_manager import load_dataset

@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name = config.app.training_data_file)
