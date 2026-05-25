'''
Reservoir-sample rows from the per-year parquet files produced by
download_data_files.py and write a single small parquet for EDA.

Each file is read in row-group batches so peak memory is proportional to the
reservoir size, not the size of any individual year file. Algorithm R
guarantees a uniform random sample across every row seen.

CLI usage:
    sec-sample
    sec-sample --n 20000 --out data/sample.parquet
    sec-sample --years 2015 2016 2017
    sec-sample --seed 123
'''

import argparse
import random
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

DEFAULT_SRC = Path('data/raw')
DEFAULT_OUT = Path(f'data/samples/{datetime.now().strftime(r"%Y%m%d_%H%M%S")}.parquet')
DEFAULT_N = 10_000


def _iter_batches(path: Path, batch_size: int = 1024):
    pf = pq.ParquetFile(path)
    yield from pf.iter_batches(batch_size=batch_size)


def sample(
    src: Path = DEFAULT_SRC,
    out: Path = DEFAULT_OUT,
    n: int = DEFAULT_N,
    years: list[str] | None = None,
    seed: int = 10,
) -> Path:
    '''
    Read parquet files under *src* and write a reservoir sample to *out*.

    Returns the output path.
    '''
    rng = random.Random(seed)
    year_set = set(years) if years else None

    parquet_files: list[Path] = sorted(
        f for f in src.glob('*.parquet')
        if year_set is None or f.stem in year_set
    )

    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found under {src} matching the given filters.')

    print(f'Sampling from {len(parquet_files)} file(s). Reservoir size: {n:,} rows')

    reservoir: list[dict] = []
    rows_seen = 0

    for pq_path in tqdm(parquet_files, unit='file'):
        for batch in _iter_batches(pq_path):
            cols = batch.to_pydict()
            col_names = list(cols.keys())
            for i in range(batch.num_rows):
                row = {col: cols[col][i] for col in col_names}
                if len(reservoir) < n:
                    reservoir.append(row)
                else:
                    j = rng.randint(0, rows_seen)
                    if j < n:
                        reservoir[j] = row
                rows_seen += 1

    if not reservoir:
        raise RuntimeError('No rows collected! Check that the source files contain data.')

    rng.shuffle(reservoir)

    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(reservoir)
    pq.write_table(table, out, compression='snappy')

    size_mb = out.stat().st_size / 1e6
    print(f'Total number of rows: {rows_seen:,} rows')
    print(f'Sample size: {len(reservoir):,} rows')
    print(f'Output: {out}  ({size_mb:.1f} MB)')
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', type=Path, default=DEFAULT_SRC,
                        help='Root dir from download_data_files.py (default: %(default)s)')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='Output parquet path (default: %(default)s)')
    parser.add_argument('--n', type=int, default=DEFAULT_N,
                        help='Reservoir / output row count (default: %(default)s)')
    parser.add_argument('--years', nargs='+',
                        help='Years to include, e.g. --years 2015 2016 2017')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()
    sample(src=args.src, out=args.out, n=args.n, years=args.years, seed=args.seed)


if __name__ == '__main__':
    main()
