from __future__ import annotations
import argparse
from dax.analysis.reporting import generate_pre_model_report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--analysis-dir',default='outputs/analysis'); p.add_argument('--output',default='outputs/analysis/reports/pre_model_analysis_report.md'); a=p.parse_args(); out=generate_pre_model_report(a.analysis_dir,a.output); print(f"Generated pre-model analysis report -> {out}")
if __name__=='__main__': main()
