from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet,Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .baselines import ConstantRegressor

def build_regressor(contract,resolved):
    fam=contract.model_family
    if fam in {'mean_baseline','median_baseline'}: return ConstantRegressor('median' if fam=='median_baseline' else 'mean')
    prep=ColumnTransformer([('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),resolved['categorical']),('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),resolved['numeric'])])
    model=HistGradientBoostingRegressor(random_state=42,**contract.hyperparameters) if fam=='hist_gradient_boosting_regressor' else (ElasticNet(random_state=42,**contract.hyperparameters) if fam=='elastic_net' else Ridge(**contract.hyperparameters))
    return Pipeline([('preprocess',prep),('model',model)])
