"""Bounded fresh-process matrix; no overlapping GPU benchmark processes."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    jobs = []
    for workload, sizes, features, degree, peers in (
        ('csr', (8192, 32768, 131072), 32, 32, ('tiga', 'pyg', 'sparse')),
        ('radius', (8192, 32768, 131072), 1, 32, ('tiga', 'pyg', 'materialized')),
        ('knn', (1024, 4096, 16384, 32768, 65536, 131072), 1, 16, ('tiga', 'pyg', 'materialized')),
    ):
        for nodes in sizes:
            for provider in peers:
                jobs.append((workload, nodes, features, degree, provider))
    random.Random(20260919).shuffle(jobs)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for workload, nodes, features, degree, provider in jobs:
        output = args.output/f'{workload}-{nodes}-{provider}.json'
        command = [sys.executable, '-m', 'benchmarks.graph_operations.paper_comparison',
                   '--workload', workload, '--nodes', str(nodes), '--features', str(features),
                   '--degree', str(degree), '--provider', provider, '--output', str(output)]
        try:
            run = subprocess.run(command, timeout=480, text=True, capture_output=True)
            record = dict(command=command, returncode=run.returncode, stdout=run.stdout, stderr=run.stderr)
        except subprocess.TimeoutExpired as error:
            record = dict(command=command, returncode='timeout', error=str(error))
        records.append(record)
        (args.output/'runs.json').write_text(json.dumps(records, indent=2)+'\n')
        print(workload, nodes, provider, record['returncode'], record.get('stdout', ''), flush=True)
        if record['returncode'] != 0:
            print(record.get('stderr', record.get('error')), flush=True)
    if any(r['returncode'] != 0 for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
