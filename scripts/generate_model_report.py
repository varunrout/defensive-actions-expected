import argparse
from dax.models.reporting import write_model_report

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/models.yaml'); p.add_argument('--output',default='outputs/models/reports/model_validation_report.md'); p.add_argument('--output-dir',default='outputs'); p.add_argument('--mlflow-enabled',action='store_true'); p.add_argument('--tracking-uri'); a=p.parse_args(argv); print(write_model_report(a.config,a.output,a.output_dir))
if __name__=='__main__': main()
