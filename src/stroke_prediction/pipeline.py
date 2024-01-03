from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engine.selection import DropFeatures
from feature_engine.encoding import OneHotEncoder
from feature_engine.wrappers import SklearnTransformerWrapper

from stroke_prediction.processing import features as pp
from stroke_prediction.config.settings import config


pipeline = Pipeline([
    ('mapper_smoke', pp.Mapper(feature=['smoking_status'],
                               mappings=config.model.smoking_mappings)),
    ('one_hot_encode', OneHotEncoder(variables=config.model.encode_vars,
                                     drop_last_binary=True)),
    ('scale', SklearnTransformerWrapper(StandardScaler())),
    ('drop_features', DropFeatures(features_to_drop=config.model.features_to_drop))
])