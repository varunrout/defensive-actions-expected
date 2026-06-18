from __future__ import annotations
import pandas as pd
from dax.models.schemas import normalise_model_schema, validate_model_dataset, dataset_fingerprint
from dax.models.feature_contracts import load_model_config, get_contracts, resolve_contract
from dax.models.leakage import scan_features
from dax.models.splits import make_grouped_folds
from dax.models.baselines import ConstantClassifier, ConstantRegressor
from dax.models.evaluation import classification_metrics, regression_metrics
from dax.models.oof_signals import build_player_signals, bootstrap_ci_by_match
from dax.models.validation import assert_oof_predictions

def fixture_df():
 return pd.DataFrame({
  'match_id':[1,1,2,2,3,3],'event_id':[f'e{i}' for i in range(6)],'player_id':[1,1,2,2,3,3],'player_name':['A','A','B','B','C','C'],'team':['T']*6,'event_type':['Pressure','Duel']*3,'action_family':['pressure','duel']*3,'phase_label':['def','mid','def','mid','def','mid'],'action_x':[10,20,30,40,50,60],'action_y':[30,35,40,45,50,55],'position_group':['DF']*6,'target_future_shot_10s':[0,1,0,0,1,0],'target_future_xg_10s':[0,.1,0,0,.2,0]})

def test_schema_and_targets():
 audit=validate_model_dataset(fixture_df()); assert audit.matches==3 and audit.shot_positive==2

def test_feature_contracts_and_leakage():
 cfg=load_model_config('configs/models.yaml'); c=get_contracts(cfg,'classification')[1]; r=resolve_contract(fixture_df(),c); assert 'phase_label' in r['final_features']
 try: scan_features(['match_id'], selected_target='target_future_shot_10s')
 except ValueError: pass
 else: raise AssertionError('leakage not detected')

def test_grouped_fold_exclusivity_and_support():
 df=fixture_df(); folds=make_grouped_folds(df,'target_future_shot_10s',n_splits=3)
 for f in folds.fold.unique():
  val=set(df.loc[folds.fold.eq(f),'match_id']); tr=set(df.loc[~folds.fold.eq(f),'match_id']); assert val.isdisjoint(tr)

def test_baselines_and_metrics():
 df=fixture_df(); clf=ConstantClassifier().fit(pd.DataFrame(index=df.index),df.target_future_shot_10s); p=clf.predict_proba(pd.DataFrame(index=df.index))[:,1]; assert classification_metrics(df.target_future_shot_10s,p)['brier_score']>=0
 reg=ConstantRegressor().fit(None,df.target_future_xg_10s); y=reg.predict(pd.DataFrame(index=df.index)); assert regression_metrics(df.target_future_xg_10s,y)['mae']>=0

def test_fingerprint_and_oof_signals(tmp_path):
 df=fixture_df(); fp=dataset_fingerprint(tmp_path/'x.parquet',df); assert fp['row_count']==6
 c=df[['event_id','match_id','player_id','player_name','team','action_family','phase_label','position_group']].copy(); c['y_true']=df.target_future_shot_10s; c['y_score']=[.1,.2,.1,.1,.2,.1]; c['fold']=[0,0,1,1,2,2]; c['model_variant']='b0'; c['model_family']='constant'; c['mlflow_run_id']='r'; c['train_match_ids']=[[2,3],[2,3],[1,3],[1,3],[1,2],[1,2]]; assert_oof_predictions(c,'y_score')
 r=c.rename(columns={'y_score':'y_pred'}).copy(); r['y_true']=df.target_future_xg_10s; r['residual']=r.y_pred-r.y_true
 cp=tmp_path/'c.parquet'; rp=tmp_path/'r.parquet'; op=tmp_path/'signals.parquet'; c.to_parquet(cp); r.to_parquet(rp); out=build_player_signals(cp,rp,op,min_actions=2); assert op.exists() and 'minimum_sample_flag' in out.columns
 assert len(bootstrap_ci_by_match(c.assign(v=1.0),'v',n=5))==3


def test_constant_estimators_are_sklearn_clone_and_joblib_compatible(tmp_path):
    import joblib
    import pandas as pd
    from sklearn.base import clone

    from dax.models.baselines import ConstantClassifier, ConstantRegressor

    x = pd.DataFrame(index=range(4))
    y_class = [0, 1, 1, 0]
    y_reg = [0.0, 0.1, 0.2, 0.0]

    classifier = clone(ConstantClassifier()).fit(x, y_class)
    regressor = clone(ConstantRegressor(stat="median")).fit(x, y_reg)

    class_path = tmp_path / "constant_classifier.joblib"
    reg_path = tmp_path / "constant_regressor.joblib"
    joblib.dump(classifier, class_path)
    joblib.dump(regressor, reg_path)

    loaded_classifier = joblib.load(class_path)
    loaded_regressor = joblib.load(reg_path)
    assert loaded_classifier.predict_proba(x).shape == (4, 2)
    assert loaded_regressor.predict(x).shape == (4,)
