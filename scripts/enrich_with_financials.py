'''
Join cleaned 10-K text (sections 1, 1A, 7) with EDGAR structured financials
and company metadata for fiscal years 2009-2020.

The script is resumable: fetched data is cached per-CIK under
data/cache/metadata/<cik>.json and data/cache/financials/<cik>.parquet.
Re-running skips already-cached companies.

Filtering rules applied before any network calls:
  - Only years 2009-2020 (XBRL-era, where financial data is reliable)
  - Only rows where ALL of section_1, section_1A, and section_7 are non-null

Joined variables
----------------
Company metadata (one row per CIK):
    company_name, sic_code, sic_description, ticker, exchange,
    state_of_incorporation, filer_category, fiscal_year_end

Annual financials (one row per CIK × fiscal_year):
    revenue, net_income, operating_income, gross_profit,
    total_assets, total_liabilities, shareholders_equity,
    eps_diluted, rd_expense, cash_and_equivalents, long_term_debt

Output:
    data/enriched/<year>.parquet — one file per year, same row order as
    data/clean/<year>.parquet (minus rows filtered out above).

Memory design
-------------
The script uses a two-pass approach to keep peak RAM at ~1 year of text data
(~4 GB) rather than all years simultaneously (~60 GB):

  Pass 1 — scan only the four lightweight columns needed for filtering/CIK
            collection (no text body). Builds a per-year set of surviving CIKs.
  Pass 2 — load one full-year parquet at a time, filter, merge with the small
            metadata/financials lookup tables, write output, then free memory.

The financial and metadata lookup tables are numeric/string scalars only
(no long text) so they stay in RAM throughout and are cheap (~100 MB total).

CLI usage:
    python scripts/enrich_with_financials.py
    python scripts/enrich_with_financials.py --src data/clean --out data/enriched
    python scripts/enrich_with_financials.py --years 2015 2016 2017
    python scripts/enrich_with_financials.py --workers 8
    python scripts/enrich_with_financials.py --identity user@example.com
'''

import argparse
import gc
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

import edgar

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SRC = Path('data/clean')
DEFAULT_OUT = Path('data/enriched')
DEFAULT_CACHE = Path('data/cache')
DEFAULT_YEARS = list(range(2009, 2025))          # 2009–2024 inclusive
DEFAULT_IDENTITY = os.getenv('EDGAR_IDENTITY', 'james.c.williams.maec@gmail.com')
DEFAULT_WORKERS = 5                              # concurrent EDGAR API calls
RATE_SLEEP = 0.12                                # ~8 req/s; EDGAR cap is ~10/s

TARGET_SECTIONS = ['section_1', 'section_1A', 'section_7']

# ── XBRL concept priority lists ───────────────────────────────────────────────
# Concepts are tried in order; the first non-null annual value wins.
CONCEPT_GROUPS: dict[str, list[str]] = {
    'revenue': [
        'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',
        'us-gaap:Revenues',
        'us-gaap:SalesRevenueNet',
        'us-gaap:SalesRevenueGoodsNet',
        'us-gaap:RevenuesNetOfInterestExpense',
    ],
    'net_income': ['us-gaap:NetIncomeLoss'],
    'operating_income': ['us-gaap:OperatingIncomeLoss'],
    'gross_profit': ['us-gaap:GrossProfit'],
    'total_assets': ['us-gaap:Assets'],
    'total_liabilities': ['us-gaap:Liabilities'],
    'shareholders_equity': [
        'us-gaap:StockholdersEquity',
        'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    ],
    'eps_diluted': ['us-gaap:EarningsPerShareDiluted'],
    'rd_expense': ['us-gaap:ResearchAndDevelopmentExpense'],
    'cash_and_equivalents': [
        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
        'us-gaap:Cash',
    ],
    'long_term_debt': [
        'us-gaap:LongTermDebt',
        'us-gaap:LongTermDebtNoncurrent',
    ],
}

# ── cache helpers ──────────────────────────────────────────────────────────────

def _meta_path(cache_dir: Path, cik: int) -> Path:
    return cache_dir / 'metadata' / f'{cik}.json'


def _fin_path(cache_dir: Path, cik: int) -> Path:
    return cache_dir / 'financials' / f'{cik}.parquet'


def _load_meta_cache(cache_dir: Path, cik: int) -> dict | None:
    p = _meta_path(cache_dir, cik)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None


def _save_meta_cache(cache_dir: Path, cik: int, data: dict) -> None:
    p = _meta_path(cache_dir, cik)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def _load_fin_cache(cache_dir: Path, cik: int) -> pd.DataFrame | None:
    p = _fin_path(cache_dir, cik)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    return None


def _save_fin_cache(cache_dir: Path, cik: int, df: pd.DataFrame) -> None:
    p = _fin_path(cache_dir, cik)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


# ── fetchers ───────────────────────────────────────────────────────────────────

def _fetch_metadata(cik: int) -> dict:
    '''Fetch company metadata via get_entity_submissions. Returns a flat dict.'''
    try:
        subs = edgar.get_entity_submissions(cik)
        tickers = subs.tickers or []
        exchanges = subs.exchanges or []
        return {
            'company_name': subs.name,
            'sic_code': subs.sic,
            'sic_description': subs.sic_description,
            'ticker': tickers[0] if tickers else None,
            'exchange': exchanges[0] if exchanges else None,
            'state_of_incorporation': subs.state_of_incorporation,
            'filer_category': subs.category,
            'fiscal_year_end': subs.fiscal_year_end,
        }
    except Exception as e:
        log.warning('metadata fetch failed for CIK %s: %s', cik, e)
        return {k: None for k in [
            'company_name', 'sic_code', 'sic_description', 'ticker',
            'exchange', 'state_of_incorporation', 'filer_category', 'fiscal_year_end',
        ]}


def _fetch_financials(cik: int) -> pd.DataFrame:
    '''
    Fetch annual XBRL financial facts for a CIK.
    Returns a DataFrame with columns: fiscal_year + one column per metric in CONCEPT_GROUPS.
    '''
    try:
        facts_obj = edgar.get_company_facts(cik)
        all_facts = facts_obj.to_dataframe()
    except Exception as e:
        log.warning('facts fetch failed for CIK %s: %s', cik, e)
        return pd.DataFrame()

    annual = all_facts[all_facts['fiscal_period'] == 'FY'].copy()
    # Free the full facts table immediately — only annual rows are needed
    del all_facts
    if annual.empty:
        return pd.DataFrame()

    # For each metric, pick the first concept group that has data and pivot by year
    rows: dict[int, dict] = {}  # fiscal_year → {metric: value}

    for metric, concepts in CONCEPT_GROUPS.items():
        for concept in concepts:
            subset = annual[annual['concept'] == concept][['fiscal_year', 'numeric_value']].copy()
            subset = subset.dropna(subset=['numeric_value'])
            if subset.empty:
                continue
            # If multiple rows per year (restated values), keep the latest period_end
            # The to_dataframe already deduplicates by picking the most recent filing;
            # drop_duplicates on fiscal_year keeps the first (highest-ranked) value.
            subset = subset.drop_duplicates('fiscal_year', keep='first')
            for _, row in subset.iterrows():
                yr = int(row['fiscal_year'])
                rows.setdefault(yr, {})[metric] = row['numeric_value']
            break  # found a concept with data for this metric — stop trying alternatives

    del annual

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'fiscal_year'
    df = df.reset_index()

    # Ensure all metric columns exist even if some CIKs had none
    for metric in CONCEPT_GROUPS:
        if metric not in df.columns:
            df[metric] = pd.NA

    return df


def _fetch_one(cik: int, cache_dir: Path) -> tuple[dict, pd.DataFrame]:
    '''Fetch or load from cache metadata + financials for a single CIK.'''
    meta = _load_meta_cache(cache_dir, cik)
    fin = _load_fin_cache(cache_dir, cik)

    if meta is None:
        time.sleep(RATE_SLEEP)
        meta = _fetch_metadata(cik)
        _save_meta_cache(cache_dir, cik, meta)

    if fin is None:
        time.sleep(RATE_SLEEP)
        fin = _fetch_financials(cik)
        _save_fin_cache(cache_dir, cik, fin)

    return meta, fin


# ── main enrichment logic ──────────────────────────────────────────────────────

def enrich(
    src_dir: Path = DEFAULT_SRC,
    out_dir: Path = DEFAULT_OUT,
    cache_dir: Path = DEFAULT_CACHE,
    years: list[int] | None = None,
    workers: int = DEFAULT_WORKERS,
) -> None:
    '''
    Read clean parquet files, filter to rows with all target sections non-null,
    fetch EDGAR metadata + financials for each unique CIK, join, and write output.

    Memory strategy: two-pass processing so only one year of text is ever in RAM.
      Pass 1 — scan lightweight columns only (cik + 3 section null flags) to
               collect unique CIKs and a per-year survivor mask.  No text bodies
               are kept in memory after each file is scanned.
      Pass 2 — load one full year at a time, filter, merge with the small
               metadata/financials tables, write, then explicitly free.
    '''
    year_set = set(years) if years else set(DEFAULT_YEARS)

    parquet_files = sorted(
        f for f in src_dir.glob('*.parquet')
        if int(f.stem) in year_set
    )
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found in {src_dir} for years {sorted(year_set)}')

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: Lightweight scan — collect CIKs, count survivors ─────────────
    # Only read the four columns needed for the non-null filter + CIK collection.
    # Text bodies (section_1 etc.) are NOT loaded here — this keeps RAM usage
    # negligible (~MB per file) during the scan.
    print('Pass 1/3 — Scanning for valid CIKs (lightweight, no text loaded) …')
    all_ciks: set[int] = set()
    # year → set of CIKs that survived the filter (used in Pass 2 for a fast
    # pre-filter before the full load, saving memory on bad rows)
    surviving_ciks_by_year: dict[int, set[int]] = {}
    year_row_counts: dict[int, int] = {}

    scan_cols = ['cik'] + TARGET_SECTIONS  # 4 columns, not the full 24

    for pq_path in tqdm(parquet_files, unit='file', desc='Scanning'):
        year = int(pq_path.stem)
        df = pd.read_parquet(pq_path, columns=scan_cols)
        mask = df['section_1'].notna() & df['section_1A'].notna() & df['section_7'].notna()
        surviving = df.loc[mask, 'cik'].dropna().astype(int)
        surviving_set = set(surviving.tolist())
        surviving_ciks_by_year[year] = surviving_set
        year_row_counts[year] = len(surviving)
        all_ciks.update(surviving_set)
        del df, surviving  # free immediately

    total_rows = sum(year_row_counts.values())
    all_ciks_sorted = sorted(all_ciks)
    print(f'  {total_rows:,} rows across {len(parquet_files)} years | '
          f'{len(all_ciks_sorted):,} unique CIKs')

    # ── Pass 2: Fetch metadata + financials (with disk cache) ─────────────────
    print(f'\nPass 2/3 — Fetching EDGAR data for {len(all_ciks_sorted):,} CIKs '
          f'({workers} workers, cache at {cache_dir}) …')

    meta_map: dict[int, dict] = {}
    fin_pieces: list[pd.DataFrame] = []

    uncached = [c for c in all_ciks_sorted
                if not _meta_path(cache_dir, c).exists() or not _fin_path(cache_dir, c).exists()]
    cached_count = len(all_ciks_sorted) - len(uncached)
    if cached_count:
        print(f'  {cached_count:,} already cached, fetching {len(uncached):,} …')

    with tqdm(total=len(all_ciks_sorted), unit='cik') as pbar:
        # Load cached entries (no network) — these are small dicts/DataFrames
        for cik in all_ciks_sorted:
            if _meta_path(cache_dir, cik).exists() and _fin_path(cache_dir, cik).exists():
                meta_map[cik] = _load_meta_cache(cache_dir, cik)  # type: ignore[assignment]
                fin_df_cik = _load_fin_cache(cache_dir, cik)
                if fin_df_cik is not None and not fin_df_cik.empty:
                    fin_df_cik = fin_df_cik.copy()
                    fin_df_cik['cik'] = int(cik)
                    fin_pieces.append(fin_df_cik)
                pbar.update(1)

        # Fetch uncached entries in parallel
        if uncached:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_fetch_one, cik, cache_dir): cik for cik in uncached}
                for future in as_completed(futures):
                    cik = futures[future]
                    try:
                        meta, fin = future.result()
                        meta_map[cik] = meta
                        if fin is not None and not fin.empty:
                            fin = fin.copy()
                            fin['cik'] = int(cik)
                            fin_pieces.append(fin)
                    except Exception as e:
                        log.error('failed for CIK %s: %s', cik, e)
                        meta_map[cik] = {k: None for k in [
                            'company_name', 'sic_code', 'sic_description', 'ticker',
                            'exchange', 'state_of_incorporation', 'filer_category', 'fiscal_year_end',
                        ]}
                    pbar.update(1)

    # Build the two small lookup tables — these have no text, so RAM is cheap
    print('\n  Building lookup tables …')
    meta_df = pd.DataFrame.from_dict(meta_map, orient='index')
    meta_df.index.name = 'cik'
    meta_df = meta_df.reset_index()
    meta_df['cik'] = meta_df['cik'].astype('int64')
    del meta_map  # individual dicts no longer needed

    if fin_pieces:
        fin_df = pd.concat(fin_pieces, ignore_index=True)
        fin_df['cik'] = fin_df['cik'].astype('int64')
        fin_df = fin_df.rename(columns={'fiscal_year': 'year'})
        fin_df['year'] = fin_df['year'].astype('int64')
    else:
        fin_df = pd.DataFrame(columns=['cik', 'year'] + list(CONCEPT_GROUPS.keys()))
    del fin_pieces

    fin_mb = fin_df.memory_usage(deep=True).sum() / 1e6
    meta_mb = meta_df.memory_usage(deep=True).sum() / 1e6
    print(f'  Lookup tables: meta={meta_mb:.0f} MB, financials={fin_mb:.0f} MB '
          f'({len(fin_df):,} rows × {len(fin_df.columns)} cols)')

    # ── Pass 3: Join and write — one year at a time ───────────────────────────
    # Peak RAM here is ~1 full year of text (~4 GB) + the small lookup tables.
    print('\nPass 3/3 — Joining and writing enriched files (one year at a time) …')

    drop_section_cols_cache: dict[str, list[str]] = {}  # cache per file schema

    for pq_path in tqdm(sorted(parquet_files), unit='year', desc='Writing'):
        year = int(pq_path.stem)
        out_path = out_dir / f'{year}.parquet'

        # Load the full year parquet (text + all columns)
        text_df = pd.read_parquet(pq_path)

        # Drop non-target section columns (keeps cik, year, filename, format + 3 sections)
        if pq_path.name not in drop_section_cols_cache:
            drop_cols = [c for c in text_df.columns
                         if c.startswith('section_') and c not in TARGET_SECTIONS]
            drop_section_cols_cache[pq_path.name] = drop_cols
        text_df = text_df.drop(columns=drop_section_cols_cache[pq_path.name])

        # Apply the filter computed in Pass 1 (avoids re-evaluating notna on text cols)
        surviving = surviving_ciks_by_year.get(year, set())
        text_df['cik'] = text_df['cik'].astype('int64')
        text_df = text_df[text_df['cik'].isin(surviving)].copy()

        if text_df.empty:
            tqdm.write(f'  {year}: no rows after filtering, skipping')
            del text_df
            continue

        # Merge with small lookup tables (no extra text data added)
        enriched = text_df.merge(meta_df, on='cik', how='left')
        del text_df  # free text memory before writing
        enriched = enriched.merge(fin_df, on=['cik', 'year'], how='left')

        table = pa.Table.from_pandas(enriched, preserve_index=False)
        pq.write_table(table, out_path, compression='snappy')
        size_mb = out_path.stat().st_size / 1e6
        tqdm.write(f'  {out_path}  ({len(enriched):,} rows, {size_mb:.1f} MB)')

        del enriched, table
        gc.collect()  # prompt Python to release memory back to OS before next year

    print('\nDone.')


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--src', type=Path, default=DEFAULT_SRC,
                        help='Directory of cleaned parquet files (default: %(default)s)')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='Output directory for enriched parquets (default: %(default)s)')
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE,
                        help='Cache directory for per-CIK metadata + financials (default: %(default)s)')
    parser.add_argument('--years', nargs='+', type=int,
                        help='Subset of years to process (default: 2009-2024)')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Concurrent EDGAR API calls (default: {DEFAULT_WORKERS}; cap ~8)')
    parser.add_argument('--identity', default=DEFAULT_IDENTITY,
                        help='Name/email for EDGAR User-Agent (or set EDGAR_IDENTITY env var)')
    args = parser.parse_args()

    if not args.identity:
        parser.error('Provide --identity or set EDGAR_IDENTITY env var (required by SEC fair-access policy)')

    edgar.set_identity(args.identity)

    enrich(
        src_dir=args.src,
        out_dir=args.out,
        cache_dir=args.cache,
        years=args.years,
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
