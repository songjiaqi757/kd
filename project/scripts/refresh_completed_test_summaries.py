#!/usr/bin/env python3
"""Refresh both result catalogs after a queued historical test batch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--completion-marker',type=Path,required=True);args=p.parse_args()
    subprocess.run([sys.executable,str(ROOT/'project/scripts/summarize_mosi_research.py'),
                    '--registry',str(ROOT/'outputs/experiments/mosi_research_v1/resource_queue/registry.json'),'--final'],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(ROOT/'project/scripts/summarize_all_dual_checkpoints.py'),
                    '--output',str(ROOT/'project/reports/all_experiments_dual_checkpoint_20261009')],check=True,cwd=ROOT)
    args.completion_marker.write_text(json.dumps({'status':'complete'})+'\n')


if __name__=='__main__':main()
