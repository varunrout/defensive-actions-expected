from __future__ import annotations
import numpy as np
import pandas as pd
class ConstantClassifier:
    def fit(self,X,y): self.p=float(np.mean(y)); return self
    def predict_proba(self,X): return np.c_[np.full(len(X),1-self.p),np.full(len(X),self.p)]
class ConstantRegressor:
    def __init__(self,stat='mean'): self.stat=stat
    def fit(self,X,y): self.value=float(np.median(y) if self.stat=='median' else np.mean(y)); return self
    def predict(self,X): return np.full(len(X),self.value)
