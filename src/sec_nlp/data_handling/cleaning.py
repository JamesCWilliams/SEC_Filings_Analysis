"""
Clean raw per-year parquet files and write cleaned copies ready for modelling.

Each year file is processed in batches to keep peak memory low. The pipeline
adds a format column, removes empty/whitespace-only entries, strips table
blocks, then nulls out short stubs.

CLI usage:
    sec-clean
    sec-clean --src data/raw --out data/clean
    sec-clean --years 2015 2016 2017
"""

import argparse
import html as _html
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

DEFAULT_SRC = Path('data/raw')
DEFAULT_OUT = Path('data/clean')

STUB_THRESHOLD = 50
TABLE_DIGIT_RATIO = 0.3
TABLE_MIN_CONSECUTIVE_SPACES = 4
TABLE_BLOCK_MIN_LINES = 3


def _iter_batches(path: Path, batch_size: int = 1024):
    pf = pq.ParquetFile(path)
    yield from pf.iter_batches(batch_size=batch_size)


def _section_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith('section_')]


def _add_format_column(df: pd.DataFrame) -> pd.DataFrame:
    df['format'] = df['filename'].str.extract(r'\.(html?|txt)$', expand=False)
    return df


def _replace_empty(df: pd.DataFrame) -> pd.DataFrame:
    for col in _section_cols(df):
        df[col] = df[col].replace('', pd.NA)
        df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
    return df


# Matches "Item 1.", "ITEM 1A.", "Item 1A. Risk Factors." etc. at the start of a section
_HEADER_RE = re.compile(r'^\s*item\s+\d+[a-z]?\.?[^\n]*\n?', re.IGNORECASE)
# Matches an all-caps title line immediately after (e.g. "RISK FACTORS\n")
_CAPS_TITLE_RE = re.compile(r'^\s*[A-Z][A-Z\s\-,&/]{2,}\n?')


def _strip_header_from_text(text) -> str | None:
    if not isinstance(text, str):
        return text
    text = _HEADER_RE.sub('', text, count=1)
    text = _CAPS_TITLE_RE.sub('', text, count=1)
    text = text.strip()
    return text if text else None


def _strip_section_headers(df: pd.DataFrame) -> pd.DataFrame:
    for col in _section_cols(df):
        df[col] = df[col].apply(_strip_header_from_text)
    return df


_WIN1252_MAP = str.maketrans({
    '\x80': '€', '\x82': '‚', '\x83': 'ƒ', '\x84': '„', '\x85': '…',
    '\x86': '†', '\x87': '‡', '\x88': 'ˆ', '\x89': '‰', '\x8a': 'Š',
    '\x8b': '‹', '\x8c': 'Œ', '\x8e': 'Ž', '\x91': '‘', '\x92': '’',
    '\x93': '“', '\x94': '”', '\x95': '•', '\x96': '–', '\x97': '—',
    '\x98': '˜', '\x99': '™', '\x9a': 'š', '\x9b': '›', '\x9c': 'œ',
    '\x9e': 'ž', '\x9f': 'Ÿ',
})
# UTF-8 smart punctuation decoded as Windows-1252, producing â€x mojibake.
# Each tuple is (bad_sequence, correct_unicode).
# translate() runs first so Latin-1–decoded C1 bytes (e.g. \x93→") are already
# promoted before we match the en/em dash patterns (which end in " / ").
_MOJIBAKE_SUBS = (
    ('â€œ', '“'),  # â€œ  → " left double quote  (0x9C)
    ('â€\x9d',   '”'),  # â€\x9d→ " right double quote (0x9D undef)
    ('â€˜', '‘'),  # â€˜  → ' left single quote   (0x98)
    ('â€™', '’'),  # â€™  → ' right single quote  (0x99)
    ('â€“', '–'),  # â€"  → – en dash            (0x93)
    ('â€”', '—'),  # â€"  → — em dash            (0x94)
    ('â€¦', '…'),  # â€¦  → … ellipsis           (0xA6)
)

_MULTI_SPACE_RE = re.compile(r'[^\S\n]+')
_MULTI_NEWLINE_RE = re.compile(r'\n{3,}')


def _decode_html(text) -> str | None:
    if not isinstance(text, str):
        return text
    text = text.translate(_WIN1252_MAP)
    for bad, good in _MOJIBAKE_SUBS:
        if bad in text:
            text = text.replace(bad, good)
    text = _html.unescape(text)
    text = text.replace('\xa0', ' ')
    return text if text else None


def _decode_html_entities(df: pd.DataFrame) -> pd.DataFrame:
    for col in _section_cols(df):
        df[col] = df[col].apply(_decode_html)
    return df


def _normalize_ws(text) -> str | None:
    if not isinstance(text, str):
        return text
    text = _MULTI_SPACE_RE.sub(' ', text)
    text = _MULTI_NEWLINE_RE.sub('\n\n', text)
    text = text.strip()
    return text if text else None


def _normalize_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    for col in _section_cols(df):
        df[col] = df[col].apply(_normalize_ws)
    return df


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if sum(c.isdigit() for c in stripped) / len(stripped) > TABLE_DIGIT_RATIO:
        return True
    if re.search(f' {{{TABLE_MIN_CONSECUTIVE_SPACES},}}', line):
        return True
    return False


def _strip_tables_from_text(text) -> str | None:
    if not isinstance(text, str):
        return text
    lines = text.splitlines()
    is_table = [_is_table_line(line) for line in lines]
    result = []
    i = 0
    while i < len(lines):
        if is_table[i]:
            j = i
            while j < len(lines) and is_table[j]:
                j += 1
            if (j - i) >= TABLE_BLOCK_MIN_LINES:
                i = j
                continue
        result.append(lines[i])
        i += 1
    cleaned = '\n'.join(result).strip()
    return cleaned if cleaned else None


def _strip_table_blocks(df: pd.DataFrame) -> pd.DataFrame:
    for col in _section_cols(df):
        df[col] = df[col].apply(_strip_tables_from_text)
    return df


def _replace_stubs(df: pd.DataFrame) -> pd.DataFrame:
    for col in _section_cols(df):
        df.loc[df[col].str.len() < STUB_THRESHOLD, col] = pd.NA
    return df


_PIPELINE = [
    _add_format_column,
    _replace_empty,
    _strip_section_headers,
    _decode_html_entities,
    _strip_table_blocks,
    _normalize_whitespace,
    _replace_stubs,
]


def _build_output_schema(path: Path) -> pa.Schema:
    src_schema = pq.read_schema(path)
    fields = [
        pa.field(f.name, pa.string() if f.name.startswith('section_') else f.type)
        for f in src_schema
    ]
    fields.append(pa.field('format', pa.string()))
    return pa.schema(fields)


def _apply_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    for step in _PIPELINE:
        df = step(df)
    return df


def clean_data(
        raw_dir: Path = DEFAULT_SRC,
        out_dir: Path = DEFAULT_OUT,
        years: list[str] | None = None,
) -> Path:
    """
    Clean parquet files under *raw_dir* and write results to *out_dir*.

    Returns the output directory path.
    """
    year_set = set(years) if years else None

    parquet_files: list[Path] = sorted(
        f for f in raw_dir.glob('*.parquet')
        if (year_set is None or f.stem in year_set) and f.stem not in ['1993', '1994', '1995']
    )

    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found under {raw_dir} matching the given filters.')

    out_dir.mkdir(parents=True, exist_ok=True)

    for pq_path in tqdm(parquet_files, unit='file'):
        out_path = out_dir / pq_path.name
        out_schema = _build_output_schema(pq_path)
        try:
            with pq.ParquetWriter(out_path, out_schema, compression='snappy') as writer:
                for batch in _iter_batches(pq_path):
                    df = batch.to_pandas()
                    df = _apply_pipeline(df)
                    table = pa.Table.from_pandas(df, preserve_index=False, schema=out_schema)
                    writer.write_table(table)
        except Exception:
            out_path.unlink(missing_ok=True)
            raise
        tqdm.write(f'{out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)')

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', type=Path, default=DEFAULT_SRC,
                        help='Root dir from download_data_files.py (default: %(default)s)')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='Output directory for clean parquets (default: %(default)s)')
    parser.add_argument('--years', nargs='+',
                        help='Years to include, e.g. --years 2015 2016 2017')
    args = parser.parse_args()
    clean_data(raw_dir=args.src, out_dir=args.out, years=args.years)


if __name__ == '__main__':
    main()
