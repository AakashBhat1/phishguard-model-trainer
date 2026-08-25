"""
PhishGuard 2.0 — Master Dataset Consolidation Engine

Merges the three source corpora into one canonical, deduplicated, leak-safe training
table:

    malicious_phish.csv               type == 'benign' -> 0, else 1
    phishing_site_urls.csv            Label == 'good'  -> 0, else 1
    PhiUSIIL_Phishing_URL_Dataset.csv label == 1       -> 0  (PhiUSIIL 1 means legitimate)

Output columns are url / label / group / source_dataset, written to CSV, Parquet and a
manifest.

--------------------------------------------------------------------------------------
Why this script audits surface form
--------------------------------------------------------------------------------------
Merging corpora that disagree on *formatting* rather than on content is how a URL model
learns a shortcut instead of a signal. This repo has already been bitten once: the two
original corpora carried an explicit scheme on 43.6% of malicious rows against 9.3% of
benign ones, so a char_wb TF-IDF fitted on the raw strings learned "has a scheme =>
phishing". normalize_url() exists to close that hole (see features.py).

PhiUSIIL carries a second, larger instance of the same defect that normalisation does
NOT close. Measured over all 235,370 of its unique rows:

    legitimate rows with a path : 0.0000   (0 of 134,850)
    phishing   rows with a path : 0.5986
    rule "path OR not-https"    : accuracy 0.9904, FPR 0.0000, FNR 0.0223

A two-predicate regex scores 99% on that corpus. Any model trained on it unmixed will
learn "bare domain => benign, has a path => phishing", report a superb PhiUSIIL number,
and then flag github.com/torvalds/linux in production.

Merging it into ~1.2M rows that *do* contain pathed benign URLs dilutes the correlation
rather than removing it. Whether the dilution is sufficient is an empirical question, so
this script measures P(phishing | has_path) per source and for the merged table, and
records both in the manifest. Read `surface_form_audit` in master_dataset_manifest.json
before you trust a training run built on this file.

--phiusiil-benign controls the trade-off:
    include      (default) keep every PhiUSIIL row; matches the plain consolidation plan
    domains-only reduce PhiUSIIL *phishing* rows to bare registered domains as well, so
                 both of its classes are bare and the within-source path correlation
                 disappears. Costs the path tokens of ~100k modern phishing URLs.
    exclude      drop the PhiUSIIL benign side; keeps its phishing URLs intact and lets
                 the benign mass come from the other two corpora.

Measured outcome of the default build: PhiUSIIL alone scores path_rule_accuracy 0.6997,
but the merged master scores 0.5264 — near chance. Diluting it against ~1.1M pathed rows
from the other two corpora is sufficient, so 'include' is safe here. Re-read the audit
if you ever change the source mix.

--------------------------------------------------------------------------------------
Why this script exposes a conflict policy
--------------------------------------------------------------------------------------
malicious_phish.csv and phishing_site_urls.csv are not independent corpora that happen to
agree. They share 439,715 normalised URLs and **disagree on 20.7% of them** (90,852),
near-symmetrically: 48,099 where the first says phishing and the second says benign,
42,753 the other way. That is mutual label noise, not evidence.

The consensus rule from the plan (mean >= 0.50) sends every one of those 1-vs-1 ties to
phishing, which manufactures ~90k phishing labels out of pure disagreement — about 22% of
the master phishing class. A model cannot exceed the quality of those labels, so the
policy is a knob rather than a constant:

    phishing (default, plan-compliant) mean >= 0.50, ties -> phishing
    benign                             mean >  0.50, ties -> benign
    drop                               remove contested URLs from the master entirely

'drop' produces a smaller but cleaner table and is the recommended setting for a run you
intend to trust; compare the two before committing to either.

Usage
    python consolidate_datasets.py
    python consolidate_datasets.py --conflict-policy drop
    python consolidate_datasets.py --phiusiil-benign domains-only
    python consolidate_datasets.py --sample 50000 --no-parquet
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from features import NORMALIZATION_VERSION, normalize_url, registered_domain

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

MASTER_CSV = 'phishguard_master_dataset.csv'
MASTER_PARQUET = 'phishguard_master_dataset.parquet'
MASTER_MANIFEST = 'master_dataset_manifest.json'

# Each source gets a distinct power of two so provenance survives the merge as a cheap
# integer OR instead of a per-group string join over ~1.2M groups.
SOURCE_BITS = {
    'malicious_phish': 1,
    'phishing_site_urls': 2,
    'phiusiil': 4,
}


def _label_malicious_phish(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower() != 'benign').astype(np.int8)


def _label_phishing_site_urls(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower() != 'good').astype(np.int8)


def _label_phiusiil(s: pd.Series) -> pd.Series:
    # PhiUSIIL encodes 1 = legitimate. Flip to the repo convention, 1 = phishing.
    return (pd.to_numeric(s, errors='coerce').fillna(1).astype(int) == 0).astype(np.int8)


SOURCE_SPECS: List[Dict[str, Any]] = [
    {
        'name': 'malicious_phish',
        'filenames': ['malicious_phish.csv'],
        'url_col': 'url',
        'label_col': 'type',
        'label_fn': _label_malicious_phish,
        'encoding': 'utf-8',
    },
    {
        'name': 'phishing_site_urls',
        'filenames': ['phishing_site_urls.csv'],
        'url_col': 'URL',
        'label_col': 'Label',
        'label_fn': _label_phishing_site_urls,
        'encoding': 'utf-8',
    },
    {
        'name': 'phiusiil',
        # Lives in test_model/ in this repo; also accept a copy beside the trainer.
        'filenames': [
            os.path.join('test_model', 'PhiUSIIL_Phishing_URL_Dataset.csv'),
            'PhiUSIIL_Phishing_URL_Dataset.csv',
        ],
        'url_col': 'URL',
        'label_col': 'label',
        'label_fn': _label_phiusiil,
        # The file carries a UTF-8 BOM, which turns the first header into '﻿FILENAME'
        # and makes a plain usecols=['URL','label'] read fail on some pandas builds.
        'encoding': 'utf-8-sig',
    },
]


def _resolve(filenames: List[str]) -> Optional[str]:
    """First existing path for a source, searched beside the trainer then at repo root."""
    for fn in filenames:
        for root in (BASE_DIR, REPO_ROOT):
            candidate = os.path.join(root, fn)
            if os.path.exists(candidate):
                return candidate
    return None


def _has_path(urls: pd.Series) -> pd.Series:
    """True when a *normalised* URL carries anything after the host.

    normalize_url() has already removed the scheme, so a surviving '/' is a real path
    separator rather than the '//' of 'https://'.
    """
    return urls.str.contains('/', regex=False)


def _surface_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """P(phishing | has_path) versus P(phishing | bare), plus the naive path rule."""
    hp = _has_path(df['url'])
    n = int(len(df))
    n_path = int(hp.sum())
    n_bare = n - n_path
    y = df['label'].to_numpy()
    hp_np = hp.to_numpy()

    phish_given_path = float(y[hp_np].mean()) if n_path else None
    phish_given_bare = float(y[~hp_np].mean()) if n_bare else None

    # Accuracy of predicting "phishing iff the URL has a path". 0.5 means the shape
    # carries no class information; anything near 1.0 means the corpus is separable on
    # formatting alone and must not be trained on as-is.
    path_rule_acc = float((hp_np.astype(np.int8) == y).mean())

    return {
        'n_rows': n,
        'pct_with_path': round(n_path / n, 6) if n else None,
        'pct_with_path_benign': round(float(hp_np[y == 0].mean()), 6) if (y == 0).any() else None,
        'pct_with_path_phishing': round(float(hp_np[y == 1].mean()), 6) if (y == 1).any() else None,
        'p_phishing_given_path': round(phish_given_path, 6) if phish_given_path is not None else None,
        'p_phishing_given_bare': round(phish_given_bare, 6) if phish_given_bare is not None else None,
        'path_rule_accuracy': round(path_rule_acc, 6),
    }


def load_source(spec: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Read one corpus down to normalised url/label/source_bit, deduped within itself."""
    path = _resolve(spec['filenames'])
    if path is None:
        print(f"  ! {spec['name']}: not found (looked for {spec['filenames']}) — skipping.")
        return None

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  • {spec['name']}: reading {os.path.relpath(path, REPO_ROOT)} ({size_mb:.1f} MB)...")

    try:
        df = pd.read_csv(
            path,
            usecols=[spec['url_col'], spec['label_col']],
            encoding=spec['encoding'],
            low_memory=False,
        )
    except ValueError:
        # Column names differ from the spec: fall back to the first two positional
        # columns, which is what the original loader did.
        df = pd.read_csv(path, encoding=spec['encoding'], low_memory=False)
        df = df.iloc[:, :2]
        df.columns = [spec['url_col'], spec['label_col']]

    df = df.dropna(subset=[spec['url_col'], spec['label_col']])
    out = pd.DataFrame({
        'url': df[spec['url_col']].astype(str).str.strip(),
        'label': spec['label_fn'](df[spec['label_col']]),
    })
    del df
    gc.collect()

    raw_rows = len(out)
    out = out[out['url'].str.len() > 3]

    # Normalise BEFORE any deduplication so http/https variants of one target collapse.
    out['url'] = out['url'].map(normalize_url)
    out = out[out['url'].str.len() > 3]

    # Collapse within-source duplicates first. Keeping this separate from the global
    # merge is what lets provenance be a plain integer sum later: after this step a
    # given url appears at most once per source.
    out = out.groupby('url', as_index=False)['label'].mean()
    out['label'] = (out['label'] >= 0.5).astype(np.int8)
    out['source_bit'] = np.int8(SOURCE_BITS[spec['name']])

    print(f"    -> {raw_rows:,} raw rows -> {len(out):,} unique normalised URLs "
          f"({int((out['label'] == 0).sum()):,} benign / {int((out['label'] == 1).sum()):,} phishing)")
    return out


def apply_phiusiil_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Reshape the PhiUSIIL frame according to --phiusiil-benign."""
    if policy == 'include':
        return df

    if policy == 'exclude':
        before = len(df)
        df = df[df['label'] == 1].copy()
        print(f"    -> policy 'exclude': dropped {before - len(df):,} benign rows, "
              f"kept {len(df):,} phishing URLs")
        return df

    if policy == 'domains-only':
        before_bare = float(_has_path(df.loc[df['label'] == 1, 'url']).mean())
        df = df.copy()
        mask = df['label'] == 1
        df.loc[mask, 'url'] = df.loc[mask, 'url'].map(registered_domain)
        df = df.groupby('url', as_index=False).agg(
            label=('label', 'mean'), source_bit=('source_bit', 'max'))
        df['label'] = (df['label'] >= 0.5).astype(np.int8)
        df['source_bit'] = df['source_bit'].astype(np.int8)
        print(f"    -> policy 'domains-only': phishing rows reduced to bare registered "
              f"domains (path share {before_bare:.4f} -> 0.0000), {len(df):,} rows remain")
        return df

    raise ValueError(f"unknown phiusiil policy: {policy}")


def consolidate(phiusiil_policy: str = 'include',
                conflict_policy: str = 'phishing',
                sample: int = 0,
                seed: int = 42) -> Dict[str, Any]:
    """Build the master table. Returns (df, manifest_fragment) via the 'df' key."""
    t0 = time.time()
    print("=" * 100)
    print(" PHISHGUARD 2.0 — MASTER DATASET CONSOLIDATION")
    print("=" * 100)
    print(f"  normalisation   : {NORMALIZATION_VERSION}")
    print(f"  phiusiil mode   : {phiusiil_policy}")
    print(f"  conflict policy : {conflict_policy}\n")

    print("[1/5] Ingesting sources")
    frames: List[pd.DataFrame] = []
    per_source: Dict[str, Any] = {}
    for spec in SOURCE_SPECS:
        frame = load_source(spec)
        if frame is None:
            continue
        if spec['name'] == 'phiusiil':
            frame = apply_phiusiil_policy(frame, phiusiil_policy)
        per_source[spec['name']] = {
            'unique_urls': int(len(frame)),
            'benign': int((frame['label'] == 0).sum()),
            'phishing': int((frame['label'] == 1).sum()),
            'surface_form': _surface_stats(frame),
        }
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            "No source datasets found. Expected malicious_phish.csv, phishing_site_urls.csv "
            "and test_model/PhiUSIIL_Phishing_URL_Dataset.csv beside the trainer or at the repo root."
        )

    print(f"\n[2/5] Cross-corpus merge & consensus labelling")
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    print(f"  • {len(df):,} rows across {len(per_source)} sources before cross-corpus dedup")

    # A url present in several sources produces several rows here. Averaging the label
    # implements the consensus rule (>= 0.5 wins, ties resolve to phishing); summing the
    # distinct source bits reconstructs provenance.
    merged = df.groupby('url', as_index=False).agg(
        label_mean=('label', 'mean'),
        source_mask=('source_bit', 'sum'),
    )
    del df
    gc.collect()

    # source_mask is a bitwise union: any value that is not an exact power of two means
    # the URL was contributed by more than one corpus.
    multi_source = ~merged['source_mask'].isin(list(SOURCE_BITS.values()))
    n_dupe = int(multi_source.sum())

    contested = (merged['label_mean'] > 0.0) & (merged['label_mean'] < 1.0)
    conflicts = int(contested.sum())

    if conflict_policy == 'drop':
        merged = merged[~contested].copy()
        merged['label'] = merged['label_mean'].round().astype(np.int8)
    elif conflict_policy == 'benign':
        merged['label'] = (merged['label_mean'] > 0.5).astype(np.int8)
    else:  # 'phishing' — the plan's rule: ties resolve upward
        merged['label'] = (merged['label_mean'] >= 0.5).astype(np.int8)
    merged.drop(columns=['label_mean'], inplace=True)

    print(f"  • {len(merged):,} unique URLs after cross-corpus dedup")
    print(f"  • {n_dupe:,} URLs seen in more than one corpus")
    print(f"  • {conflicts:,} URLs had disagreeing labels across corpora "
          f"({conflicts / max(1, conflicts + len(merged)) * 100:.1f}% of contested+kept)")
    if conflict_policy == 'drop':
        print(f"    -> policy 'drop': removed, {len(merged):,} uncontested URLs remain")
    else:
        print(f"    -> policy '{conflict_policy}': kept, ties resolved to {conflict_policy}")
        print(f"       NOTE: these are coin-flip labels. They are ~"
              f"{conflicts / max(1, int((merged['label'] == 1).sum())) * 100:.0f}% of the "
              f"phishing class and cap how far training can get.")

    bit_to_name = {v: k for k, v in SOURCE_BITS.items()}

    def mask_name(m: int) -> str:
        return '+'.join(sorted(name for bit, name in bit_to_name.items() if m & bit))

    unique_masks = sorted(merged['source_mask'].unique().tolist())
    merged['source_dataset'] = merged['source_mask'].map({m: mask_name(int(m)) for m in unique_masks})
    merged.drop(columns=['source_mask'], inplace=True)

    print(f"\n[3/5] Registered-domain group tagging (leak-safe split key)")
    merged['group'] = merged['url'].map(registered_domain)

    # A URL that yields no host at all ('?guid=windows updates manager', binary garbage)
    # has no valid split key. Left in, every such row collapses into one giant '' group
    # that GroupShuffleSplit would then dump wholly into a single partition.
    empty_groups = int((merged['group'].str.len() == 0).sum())
    if empty_groups:
        merged = merged[merged['group'].str.len() > 0].copy()
        print(f"  • dropped {empty_groups} row(s) with no derivable host")

    n_groups = int(merged['group'].nunique())
    print(f"  • {n_groups:,} unique registered domains over {len(merged):,} URLs "
          f"({len(merged) / max(1, n_groups):.1f} URLs per domain)")

    if sample and sample > 0 and len(merged) > sample:
        print(f"\n  • --sample: downsampling to {sample:,} rows, label-stratified")
        merged = (merged.groupby('label', group_keys=False, sort=False)
                  .sample(frac=sample / len(merged), random_state=seed)
                  .reset_index(drop=True))

    merged = merged[['url', 'label', 'group', 'source_dataset']]
    merged = merged.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"\n[4/5] Surface-form audit (see module docstring)")
    global_stats = _surface_stats(merged)
    print(f"  {'source':<28} {'rows':>10}  {'path|benign':>12} {'path|phish':>11} {'path-rule acc':>14}")
    print("  " + "-" * 80)
    for name, s in per_source.items():
        sf = s['surface_form']
        print(f"  {name:<28} {sf['n_rows']:>10,}  {_fmt(sf['pct_with_path_benign']):>12} "
              f"{_fmt(sf['pct_with_path_phishing']):>11} {sf['path_rule_accuracy']:>14.4f}")
    print("  " + "-" * 80)
    print(f"  {'MASTER (merged)':<28} {global_stats['n_rows']:>10,}  "
          f"{_fmt(global_stats['pct_with_path_benign']):>12} "
          f"{_fmt(global_stats['pct_with_path_phishing']):>11} "
          f"{global_stats['path_rule_accuracy']:>14.4f}")
    print("\n  path-rule accuracy near 0.50 = surface form carries no class signal (good).")
    print("  Near 1.00 = the corpus is separable on formatting alone; do not train on it as-is.")

    benign = int((merged['label'] == 0).sum())
    phishing = int((merged['label'] == 1).sum())

    manifest = {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'url_normalization': NORMALIZATION_VERSION,
        'phiusiil_benign_policy': phiusiil_policy,
        'conflict_policy': conflict_policy,
        'schema': {
            'url': 'normalised URL string (scheme, leading www. and trailing / removed)',
            'label': '1 = phishing / malicious, 0 = benign',
            'group': 'approximate eTLD+1, the GroupShuffleSplit key that prevents host leakage',
            'source_dataset': "originating corpora, '+'-joined when a URL appears in several",
        },
        'totals': {
            'unique_urls': int(len(merged)),
            'benign': benign,
            'phishing': phishing,
            'benign_pct': round(benign / max(1, len(merged)), 4),
            'phishing_pct': round(phishing / max(1, len(merged)), 4),
            'unique_registered_domains': n_groups,
            'urls_per_domain': round(len(merged) / max(1, n_groups), 3),
            'cross_corpus_duplicates': n_dupe,
            'label_conflicts': conflicts,
            'label_conflict_handling': conflict_policy,
            'rows_dropped_no_host': empty_groups,
        },
        'per_source': per_source,
        'source_dataset_counts': {k: int(v) for k, v in
                                  merged['source_dataset'].value_counts().items()},
        'surface_form_audit': {
            'note': ('path_rule_accuracy is the accuracy of predicting "phishing iff the '
                     'normalised URL contains a path". 0.5 means no shortcut; values near '
                     '1.0 mean the corpus is separable on formatting alone.'),
            'master': global_stats,
            'per_source': {k: v['surface_form'] for k, v in per_source.items()},
        },
        'build_seconds': round(time.time() - t0, 2),
    }
    return {'df': merged, 'manifest': manifest}


def _fmt(v: Optional[float]) -> str:
    return '   n/a' if v is None else f"{v:.4f}"


def export(df: pd.DataFrame, manifest: Dict[str, Any], out_dir: str,
           write_parquet: bool = True) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[5/5] Exporting to {out_dir}")

    csv_path = os.path.join(out_dir, MASTER_CSV)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    csv_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"  • {MASTER_CSV}  ({csv_mb:.1f} MB)")
    manifest.setdefault('outputs', {})[MASTER_CSV] = {'size_mb': round(csv_mb, 2)}

    if write_parquet:
        parquet_path = os.path.join(out_dir, MASTER_PARQUET)
        try:
            df.to_parquet(parquet_path, index=False, compression='snappy')
            pq_mb = os.path.getsize(parquet_path) / (1024 * 1024)
            print(f"  • {MASTER_PARQUET}  ({pq_mb:.1f} MB)")
            manifest['outputs'][MASTER_PARQUET] = {'size_mb': round(pq_mb, 2)}
        except ImportError as e:
            print(f"  ! Parquet skipped: {e}")
            print("    Install an engine to enable it:  pip install pyarrow")
            manifest['outputs'][MASTER_PARQUET] = {'skipped': 'no parquet engine installed'}

    manifest_path = os.path.join(out_dir, MASTER_MANIFEST)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"  • {MASTER_MANIFEST}")
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Consolidate the three phishing corpora into one master dataset.")
    ap.add_argument('--out-dir', default=BASE_DIR,
                    help="Directory for the master dataset files (default: the training pack).")
    ap.add_argument('--phiusiil-benign', choices=['include', 'domains-only', 'exclude'],
                    default='include',
                    help="How to handle PhiUSIIL's formatting artifact. See the module docstring.")
    ap.add_argument('--conflict-policy', choices=['phishing', 'benign', 'drop'],
                    default='phishing',
                    help="Resolution for URLs the corpora label differently. 'phishing' is "
                         "the plan's mean>=0.50 rule; 'drop' is cleaner. See the docstring.")
    ap.add_argument('--sample', type=int, default=0,
                    help="Emit only N label-stratified rows (smoke-test builds).")
    ap.add_argument('--no-parquet', action='store_true', help="Skip the Parquet export.")
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args(argv)

    result = consolidate(phiusiil_policy=args.phiusiil_benign,
                         conflict_policy=args.conflict_policy,
                         sample=args.sample, seed=args.seed)
    manifest = export(result['df'], result['manifest'], args.out_dir,
                      write_parquet=not args.no_parquet)

    t = manifest['totals']
    print("\n" + "=" * 100)
    print(f" MASTER DATASET READY — {t['unique_urls']:,} URLs "
          f"({t['benign_pct'] * 100:.1f}% benign / {t['phishing_pct'] * 100:.1f}% phishing) "
          f"across {t['unique_registered_domains']:,} domains")
    print(f" built in {manifest['build_seconds']}s")
    print("=" * 100)
    return 0


if __name__ == '__main__':
    sys.exit(main())
