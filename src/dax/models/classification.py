from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .baselines import ConstantClassifier

def build_classifier(contract,resolved):
    fam=contract.model_family
    if fam=='constant_rate': return ConstantClassifier()
    prep=ColumnTransformer([('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),resolved['categorical']),('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),resolved['numeric'])])
    if fam=='hist_gradient_boosting_classifier': model=HistGradientBoostingClassifier(random_state=42,**contract.hyperparameters)
    else: model=LogisticRegression(random_state=42,**contract.hyperparameters)
    return Pipeline([('preprocess',prep),('model',model)])
