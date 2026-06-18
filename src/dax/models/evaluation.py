from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,roc_auc_score,mean_absolute_error,mean_squared_error,r2_score

def expected_calibration_error(y,p,n_bins=10):
    y=np.asarray(y); p=np.asarray(p); bins=np.linspace(0,1,n_bins+1); e=0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(p>=lo)&(p<(hi if hi<1 else hi+1e-12))
        if m.any(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(e)
def calibration_slope_intercept(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6); logit=np.log(p/(1-p))
    if len(np.unique(y))<2: return float('nan'),float('nan')
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression().fit(logit.reshape(-1,1),y)
    return float(lr.coef_[0][0]),float(lr.intercept_[0])
def classification_metrics(y,p):
    y=np.asarray(y); p=np.clip(np.asarray(p),1e-6,1-1e-6); slope,inter=calibration_slope_intercept(y,p)
    return {'log_loss':float(log_loss(y,p,labels=[0,1])),'brier_score':float(brier_score_loss(y,p)),'average_precision':float(average_precision_score(y,p)) if y.sum()>0 else float('nan'),'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float('nan'),'calibration_slope':slope,'calibration_intercept':inter,'expected_calibration_error':expected_calibration_error(y,p),'positive_rate':float(np.mean(y))}
def regression_metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); nz=y>0; z=~nz
    sp=spearmanr(y,p).statistic if len(y)>1 else np.nan
    return {'mae':float(mean_absolute_error(y,p)),'rmse':float(mean_squared_error(y,p)**0.5),'r2':float(r2_score(y,p)) if len(y)>1 else float('nan'),'spearman':float(sp) if not np.isnan(sp) else float('nan'),'mean_prediction':float(p.mean()),'mean_observed':float(y.mean()),'prediction_bias':float((p-y).mean()),'zero_target_mae':float(mean_absolute_error(y[z],p[z])) if z.any() else float('nan'),'nonzero_target_mae':float(mean_absolute_error(y[nz],p[nz])) if nz.any() else float('nan'),'nonzero_target_rmse':float(mean_squared_error(y[nz],p[nz])**0.5) if nz.any() else float('nan')}
