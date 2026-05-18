'''
Download eloukas/edgar-corpus and save one parquet file per year.

The train/validate/test source splits are merged into a single file since the
original split boundaries are arbitrary. Custom splits can be applied later.

Output layout:
    data/raw/1993.parquet
    data/raw/1994.parquet
    ...

Already-downloaded files are skipped so the script is safe to re-run or
interrupt (Ctrl+C) and resume.

Usage:
    python scripts/download_data_files.py
    python scripts/download_data_files.py --out data/raw
    python scripts/download_data_files.py --years 2018 2019 2020
'''

import argparse
import signal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm.auto import tqdm

REPO_ID = 'eloukas/edgar-corpus'
SRC_SPLITS = {'train', 'validate', 'test'}
DEFAULT_OUT = Path('data/raw')


def group_by_year(all_files: list[str]) -> dict[str, list[str]]:
    by_year: dict[str, list[str]] = {}
    for path in all_files:
        parts = path.split('/')
        if len(parts) == 2:
            year, name = parts
            stem = name.removesuffix('.jsonl')
            if stem in SRC_SPLITS:
                by_year.setdefault(year, []).append(path)
    return dict(sorted(by_year.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--years', nargs='+',
                        help='Limit to specific years, e.g. --years 2018 2019')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    interrupted = False

    def _handle_sigint(_sig, _frame):
        nonlocal interrupted
        print('\nInterrupted: current file will finish writing, then program will exit.')
        interrupted = True

    signal.signal(signal.SIGINT, _handle_sigint)

    print(f'Listing files in {REPO_ID} ...')
    all_files = list(list_repo_files(REPO_ID, repo_type='dataset'))
    by_year = group_by_year(all_files)

    if args.years:
        year_set = set(args.years)
        by_year = {y: files for y, files in by_year.items() if y in year_set}

    for year, repo_files in tqdm(by_year.items(), unit='year'):
        if interrupted:
            break

        out_path = args.out / f'{year}.parquet'
        if out_path.exists():
            tqdm.write(f'skip {out_path} (already exists)')
            continue

        frames = []
        for repo_file in repo_files:
            local = hf_hub_download(repo_id=REPO_ID, filename=repo_file, repo_type='dataset')
            frames.append(pd.read_json(local, lines=True))

        df = pd.concat(frames, ignore_index=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, out_path, compression='snappy')
        tqdm.write(f'{out_path}  {len(df):,} rows  {out_path.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
