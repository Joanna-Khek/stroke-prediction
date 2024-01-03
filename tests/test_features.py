import pandas as pd

from stroke_prediction.config.settings import config
from stroke_prediction.processing.features import Mapper

def test_mapper_transformer():
    data = pd.DataFrame({'smoking_status': ['formerly smoked', 'smokes', 'never smoked', 'Unknown']})
    transformer = Mapper(feature=['smoking_status'],
                         mappings=config.model.smoking_mappings)
    subject = transformer.fit_transform(data)
    allowed_categories = ['smokes', 'never_smoked', 'Unknown']
    
    assert subject.smoking_status.isin(allowed_categories).all() == True