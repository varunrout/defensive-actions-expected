import argparse
from dax.models.oof_signals import build_player_signals

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--classification-oof',required=True); p.add_argument('--regression-oof',required=True); p.add_argument('--output',required=True); p.add_argument('--config',default='configs/models.yaml'); p.add_argument('--mlflow-enabled',action='store_true'); p.add_argument('--tracking-uri'); a=p.parse_args(argv); print(build_player_signals(a.classification_oof,a.regression_oof,a.output).shape)
if __name__=='__main__': main()
