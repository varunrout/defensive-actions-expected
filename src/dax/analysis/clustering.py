from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
ID_COLS={"player_id","player_name","team","minimum_sample_flag"}
def prepare_clustering_matrix(df: pd.DataFrame, min_actions:int=30) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    eligible=df[df.get("total_actions",0)>=min_actions].copy(); ids=eligible[[c for c in ["player_id","player_name","team","total_actions"] if c in eligible]]
    feats=eligible.select_dtypes(include="number").drop(columns=[c for c in eligible.select_dtypes(include="number").columns if c in ID_COLS or c.endswith("_denominator")], errors="ignore")
    nun=feats.nunique(dropna=True); constant=nun[nun<=1].index.tolist(); feats=feats.drop(columns=constant)
    imputed=pd.DataFrame(SimpleImputer(strategy="median").fit_transform(feats), columns=feats.columns, index=feats.index) if len(feats.columns) else pd.DataFrame(index=feats.index)
    scaled=pd.DataFrame(StandardScaler().fit_transform(imputed), columns=imputed.columns, index=imputed.index) if len(imputed.columns) else imputed
    return ids.reset_index(drop=True).join(scaled.reset_index(drop=True)), ids.reset_index(drop=True).join(imputed.reset_index(drop=True)), {"minimum_actions":min_actions,"features":list(scaled.columns),"constant_features_removed":constant,"missing_value_policy":"median_imputation","scaling":"standard"}
def _scores(x, labels):
    k=len(set(labels))
    if len(x)<=k or k<2: return {"silhouette":np.nan,"calinski_harabasz":np.nan,"davies_bouldin":np.nan}
    return {"silhouette":float(silhouette_score(x,labels)),"calinski_harabasz":float(calinski_harabasz_score(x,labels)),"davies_bouldin":float(davies_bouldin_score(x,labels))}
def run_clustering(matrix: pd.DataFrame, candidates=(2,3,4), seed:int=42) -> dict[str,pd.DataFrame]:
    idc=[c for c in ["player_id","player_name","team","total_actions"] if c in matrix]; x=matrix.drop(columns=idc).fillna(0)
    evals=[]; assigns=matrix[idc].copy()
    best=None; best_score=-999
    max_k=max(1, min(len(x)-1, max(candidates) if candidates else 2))
    for k in [c for c in candidates if 1<c<=max_k]:
        models={"kmeans":KMeans(n_clusters=k,random_state=seed,n_init=10),"hierarchical":AgglomerativeClustering(n_clusters=k),"gmm":GaussianMixture(n_components=k,random_state=seed)}
        for name,m in models.items():
            labels=m.fit_predict(x) if name!="gmm" else m.fit(x).predict(x); sc=_scores(x,labels); sizes=pd.Series(labels).value_counts(); evals.append({"method":name,"clusters":k,**sc,"min_cluster_size":int(sizes.min()),"max_cluster_size":int(sizes.max()),"size_balance":float(sizes.min()/sizes.max())})
            if name=="kmeans" and sc["silhouette"]>best_score: best=(k,labels); best_score=sc["silhouette"]
    if best is None: assigns["cluster"]=-1
    else: assigns["cluster"]=best[1]
    profiles=assigns.groupby("cluster").agg(players=("cluster","size"),mean_total_actions=("total_actions","mean" if "total_actions" in assigns else "size")).reset_index()
    pca_scores=pd.DataFrame(); loadings=pd.DataFrame(); explained=pd.DataFrame()
    if len(x.columns) and len(x)>=2:
        p=PCA(n_components=min(2,len(x.columns),len(x))).fit(x); pca_scores=assigns[idc].join(pd.DataFrame(p.transform(x),columns=["pc1","pc2"][:p.n_components_])); loadings=pd.DataFrame(p.components_.T,index=x.columns).reset_index(names="feature"); explained=pd.DataFrame({"component":[f"pc{i+1}" for i in range(p.n_components_)],"explained_variance_ratio":p.explained_variance_ratio_})
    return {"cluster_evaluation":pd.DataFrame(evals),"player_clusters":assigns,"cluster_profiles":profiles,"cluster_centroids":matrix.drop(columns=idc).join(assigns["cluster"]).groupby("cluster").mean(numeric_only=True).reset_index(),"cluster_stability":pd.DataFrame([{"method":"resample_placeholder","stability_score":np.nan,"note":"Bootstrap stability hook; use evaluation sensitivity tables for selection."}]),"pca_scores":pca_scores,"pca_loadings":loadings,"pca_explained_variance":explained}
def write_clustering_outputs(tables, outdir, metadata):
    p=Path(outdir); p.mkdir(parents=True,exist_ok=True)
    for n,t in tables.items(): t.to_parquet(p/f"{n}.parquet",index=False); t.to_csv(p/f"{n}.csv",index=False)
    (p/"preprocessing_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
