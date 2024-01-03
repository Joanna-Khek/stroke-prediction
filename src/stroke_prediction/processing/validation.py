from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ValidationError

from stroke_prediction.config.settings import config

class StrokeDataInputSchema(BaseModel):
    id: int
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str
    stroke: int

class MultipleStrokeDataInput(BaseModel):
    inputs: List[StrokeDataInputSchema]


def validate_inputs(input_data: pd.DataFrame):
    errors = None
    try:
        MultipleStrokeDataInput(inputs=input_data.to_dict(orient="records"))
    except ValidationError as error:
        error = error.json()
    return input_data, errors
