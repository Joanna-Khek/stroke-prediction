from pathlib import Path
import yaml
from typing import Dict, List
from pydantic import BaseModel

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_ROOT.parent.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config" / "main.yaml"
BEST_PARAM_FILE_PATH = PACKAGE_ROOT / "config" / "best_param.yaml"
DATASET_DIR = ROOT / "data"
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "models"

class AppConfig(BaseModel):

    package_name: str
    training_data_file: str
    pipeline_name: str
    pipeline_save_file: str

class ModelConfig(BaseModel):

    target: str
    features: List[str]
    random_state: int
    test_size: float
    encode_vars: List[str]
    features_to_drop: List[str]
    smoking_mappings: Dict[str, str]
    selected_model_name: str
    selected_model_integer_list: List[str]

class Config(BaseModel):
    
    app: AppConfig
    model: ModelConfig

def fetch_config_from_yaml(file_path) -> yaml:
    """Parse YAML containing the package configuration"""

    with open(file_path, 'r') as yaml_file:
        parsed_config = yaml.safe_load(yaml_file)
        return parsed_config


def create_and_validate_config(file_path: Path, parsed_config: yaml = None):
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml(file_path)
    
    _config = Config(**parsed_config)
    return _config

config = create_and_validate_config(file_path=CONFIG_FILE_PATH)
best_param_config = fetch_config_from_yaml(file_path=BEST_PARAM_FILE_PATH)