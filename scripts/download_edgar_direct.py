'''
Download 10-K filings directly from EDGAR for years not covered by the
eloukas/edgar-corpus HuggingFace dataset (2021 onwards).

Output layout matches download_data_files.py exactly:
    data/raw/2021.parquet
    data/raw/2022.parquet
    ...

Resumable: completed filings are tracked in data/raw/.staging/<year>/done.txt.
Interrupting and re-running skips already-processed filings within a year.
Only finishes writing data/raw/<year>.parquet once the full year is complete.

Usage:
    python scripts/download_edgar_direct.py
    python scripts/download_edgar_direct.py --out data/raw
    python scripts/download_edgar_direct.py --years 2021 2022
    python scripts/download_edgar_direct.py --identity user@example.com
'''

import argparse
import gc
import logging
import os
import shutil
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from edgar import get_filings, set_identity
from tqdm.auto import tqdm

DEFAULT_OUT = Path('data/raw')
DEFAULT_YEARS = [2021, 2022, 2023, 2024]
DEFAULT_IDENTITY = os.getenv('EDGAR_IDENTITY', 'james.c.williams.maec@gmail.com')
DEFAULT_WORKERS = 5   # concurrent filing downloads; EDGAR allows ~10 req/s
CHUNK_SIZE = 200      # rows per staging parquet chunk
BATCH_SIZE = 100      # filings submitted to thread pool at once; caps live futures in memory

# Columns match the HuggingFace-sourced parquets, with section_1C added
# (SEC cybersecurity disclosure rule, effective for FY2023+).
SECTION_COLS = [
    'section_1', 'section_1A', 'section_1B', 'section_1C',
    'section_2', 'section_3', 'section_4', 'section_5', 'section_6',
    'section_7', 'section_7A', 'section_8', 'section_9', 'section_9A',
    'section_9B', 'section_10', 'section_11', 'section_12', 'section_13',
    'section_14', 'section_15',
]

# Maps parquet column name → edgartools item key
_COL_TO_ITEM = {col: 'Item ' + col.removeprefix('section_').replace('_', '.') for col in SECTION_COLS}
# e.g. section_1A → 'Item 1A', section_7A → 'Item 7A'

SCHEMA = pa.schema([
    pa.field('filename', pa.string()),
    pa.field('cik', pa.int64()),
    pa.field('year', pa.int64()),
    *[pa.field(col, pa.string()) for col in SECTION_COLS],
])

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)


def _fiscal_year_quarters(year: int) -> list[tuple[int, int]]:
    # Fiscal year ends can fall in any month of `year`; calendar-year companies
    # (the majority) file in Q1 of year+1. Spanning both covers all cases.
    return [(year, q) for q in range(1, 5)] + [(year + 1, 1), (year + 1, 2)]


def _extract_row(filing, target_year: int) -> dict | None:
    try:
        tenk = filing.obj()
        if tenk is None:
            return None

        # Verify fiscal year from the parsed TenK — no extra network call since
        # we already downloaded the filing. Avoids the expensive sgml() call that
        # filing.period_of_report triggers (it downloads the full SGML bundle).
        period = tenk.period_of_report
        if not period or int(str(period)[:4]) != target_year:
            return None  # off-year filing (e.g. FY2020 late filer in Q1 2022)

        row: dict = {
            'filename': f'{filing.cik}_{target_year}.htm',
            'cik': int(filing.cik),
            'year': target_year,
        }
        available = set(tenk.items) if tenk.items else set()
        for col, item_key in _COL_TO_ITEM.items():
            if item_key not in available:
                row[col] = None
                continue
            try:
                text = tenk[item_key]
                row[col] = str(text) if text else None
            except Exception:
                row[col] = None
        return row
    except Exception as e:
        log.warning('skipping %s: %s', filing.accession_no, e)
        return None


def _done_path(staging_dir: Path) -> Path:
    return staging_dir / 'done.txt'


def _load_done(staging_dir: Path) -> set[str]:
    p = _done_path(staging_dir)
    return set(p.read_text().splitlines()) if p.exists() else set()


def _mark_done(staging_dir: Path, accession_no: str) -> None:
    with open(_done_path(staging_dir), 'a') as f:
        f.write(accession_no + '\n')


def _write_chunk(staging_dir: Path, rows: list[dict], chunk_idx: int) -> None:
    tbl = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(tbl, staging_dir / f'chunk_{chunk_idx:05d}.parquet', compression='snappy')


def _merge_chunks(staging_dir: Path, out_path: Path) -> None:
    chunks = sorted(staging_dir.glob('chunk_*.parquet'))
    if not chunks:
        raise RuntimeError(f'No chunks in {staging_dir} — no filings were downloaded.')
    merged = pa.concat_tables([pq.read_table(c) for c in chunks])
    pq.write_table(merged, out_path, compression='snappy')


def download_year(year: int, out_dir: Path, interrupted_flag: list[bool], workers: int = DEFAULT_WORKERS) -> None:
    out_path = out_dir / f'{year}.parquet'
    if out_path.exists():
        tqdm.write(f'skip {out_path} (already exists)')
        return

    staging_dir = out_dir / '.staging' / str(year)
    staging_dir.mkdir(parents=True, exist_ok=True)

    done = _load_done(staging_dir)
    chunk_idx = len(list(staging_dir.glob('chunk_*.parquet')))
    buffer: list[dict] = []

    quarters = _fiscal_year_quarters(year)
    print(f'\n=== FY{year} ({len(quarters)} quarters, {len(done)} filings already done) ===')

    for q_year, quarter in quarters:
        if interrupted_flag[0]:
            break

        print(f'  Loading index {q_year} Q{quarter} ...', end='', flush=True)
        try:
            filings = get_filings(form='10-K', year=q_year, quarter=quarter)
        except Exception as e:
            print(f' failed: {e}')
            continue

        if not filings:
            print(' no filings returned')
            continue

        # Do NOT access f.period_of_report here — it downloads the full SGML
        # bundle for every filing just to read the header. Fiscal year filtering
        # happens inside _extract_row after the TenK is already parsed.
        in_scope = [f for f in filings if f.accession_no not in done]
        already_done = len(filings) - len(in_scope)
        print(f' {len(filings)} filings, {len(in_scope)} to download, {already_done} already done')

        if not in_scope:
            continue

        with tqdm(total=len(in_scope), desc=f'  {q_year} Q{quarter}',
                  unit='filing', dynamic_ncols=True, leave=True) as pbar:
            for batch_start in range(0, len(in_scope), BATCH_SIZE):
                if interrupted_flag[0]:
                    break
                batch = in_scope[batch_start:batch_start + BATCH_SIZE]

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    # Submit only this batch — caps live futures (and their
                    # result dicts) to BATCH_SIZE rather than the whole quarter.
                    futures: dict = {pool.submit(_extract_row, f, year): f for f in batch}

                    for future in as_completed(futures):
                        if interrupted_flag[0]:
                            pool.shutdown(wait=False, cancel_futures=True)
                            break

                        filing = futures.pop(future)
                        row = future.result()
                        # Explicitly clear the cached SGML (10–50 MB per filing)
                        # that edgartools stores on the filing object. Without
                        # this the data lives until the batch list is replaced.
                        filing._sgml = None
                        filing._filing_homepage = None
                        if row is not None:
                            buffer.append(row)
                        done.add(filing.accession_no)
                        _mark_done(staging_dir, filing.accession_no)
                        pbar.update(1)

                        if len(buffer) >= CHUNK_SIZE:
                            _write_chunk(staging_dir, buffer, chunk_idx)
                            chunk_idx += 1
                            buffer = []

                # Force a GC cycle between batches to reclaim BeautifulSoup
                # parse trees (which have circular refs and aren't freed by
                # reference counting alone).
                gc.collect()

    # Flush remaining rows
    if buffer:
        _write_chunk(staging_dir, buffer, chunk_idx)

    if interrupted_flag[0]:
        tqdm.write(f'  Interrupted — progress saved in {staging_dir}')
        return

    # Merge and finalise
    _merge_chunks(staging_dir, out_path)
    shutil.rmtree(staging_dir)
    size_mb = out_path.stat().st_size / 1e6
    tqdm.write(f'{out_path}  ({size_mb:.1f} MB)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--years', nargs='+', type=int, default=DEFAULT_YEARS,
                        help='Fiscal years to download (default: 2021 2022 2023 2024)')
    parser.add_argument('--identity', default=DEFAULT_IDENTITY,
                        help='Name/email for EDGAR User-Agent (or set EDGAR_IDENTITY env var)')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Concurrent filing downloads (default: {DEFAULT_WORKERS}; EDGAR cap ~10 req/s)')
    args = parser.parse_args()

    if not args.identity:
        parser.error('Provide --identity or set EDGAR_IDENTITY env var (required by SEC fair-access policy)')

    set_identity(args.identity)
    args.out.mkdir(parents=True, exist_ok=True)

    interrupted = [False]

    def _handle_sigint(_sig, _frame):
        if interrupted[0]:
            print('\nForce quitting.')
            raise SystemExit(1)
        print('\nInterrupt received — finishing current filing then stopping. Press Ctrl-C again to force quit.')
        interrupted[0] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    for year in sorted(args.years):
        if interrupted[0]:
            break
        download_year(year, args.out, interrupted, workers=args.workers)


if __name__ == '__main__':
    main()
