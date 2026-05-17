"""
KKBox churn data: build an eligible user cohort and extract a subsample.

Eligibility:
  1. registration_init_time in [obs_start, obs_end] (members file)
  2. at least one transaction; first_transaction_date = min(transaction_date)

Then random sample `n_sample` users and write filtered tables without
loading full files into memory.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import py7zr


@dataclass
class KKBoxSubsampleConfig:
    """Paths and parameters — fill in before running."""

    raw_dir: Path
    out_dir: Path

    members: str = "members_v3.csv"
    transactions: str = "transactions.csv"
    transactions_v2: Optional[str] = "transactions_v2.csv"
    user_logs: str = "user_logs.csv"
    user_logs_v2: Optional[str] = "user_logs_v2.csv"
    merge_outputs: bool = True
    keep_partial_extracts: bool = False

    # Observation window for account creation (YYYYMMDD integers)
    obs_start: int = 20150101
    obs_end: int = 20170228

    n_sample: int = 1000
    random_state: int = 42
    chunksize: int = 500_000

    # Plausible registration dates (drop Kaggle outliers)
    reg_min: int = 20000101
    reg_max: int = 20170228  # same as obs_end

    files_to_extract: tuple[str, ...] = ("members", "transactions", "user_logs")

    def path(self, name: str) -> Path:
        return self.raw_dir / name

    def resolve_existing(self, name: Optional[str]) -> Optional[Path]:
        if name is None:
            return None
        p = self.path(name)
        if p.exists():
            return p
        alt = self.raw_dir / f"{name}.7z"
        if alt.exists():
            return alt
        return None


def _is_7z(path: Path) -> bool:
    return path.name.endswith(".7z")


def _csv_path_for_archive(archive: Path) -> Path:
    """Path to the CSV inside a .csv.7z archive (e.g. members_v3.csv.7z -> members_v3.csv)."""
    return archive.with_name(archive.name.removesuffix(".7z"))


def ensure_csv(path: Path) -> Path:
    """Return a readable .csv path, extracting from .7z once with py7zr if needed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not _is_7z(path):
        return path

    csv_path = _csv_path_for_archive(path)
    if csv_path.exists():
        return csv_path

    print(f"Extracting {path.name} -> {csv_path.name} (one-time)...")
    with py7zr.SevenZipFile(path, mode="r") as z:
        inner_names = [n for n in z.getnames() if n.endswith(".csv")]
        if len(inner_names) != 1:
            raise RuntimeError(f"Expected one CSV in {path}, found {inner_names}")
        inner_name = inner_names[0]
        z.extract(path=path.parent, targets=[inner_name])

    extracted = path.parent / inner_name
    if extracted.resolve() != csv_path.resolve():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(csv_path))
        # remove empty parent dirs left by nested archives (e.g. data/churn_comp_refresh/)
        parent = extracted.parent
        while parent != path.parent and parent != parent.parent:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    if not csv_path.exists():
        raise FileNotFoundError(f"Expected {csv_path} after extracting {path}")
    return csv_path


def iter_csv_chunks(
    path: Path,
    *,
    chunksize: int,
    usecols: Optional[list[str]] = None,
) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks from a .csv or .csv.7z file."""
    csv_path = ensure_csv(Path(path))
    read_kwargs = dict(chunksize=chunksize, low_memory=False)
    if usecols is not None:
        read_kwargs["usecols"] = usecols
    yield from pd.read_csv(csv_path, **read_kwargs)


def scan_registered_in_observation(
    members_path: Path,
    *,
    obs_start: int,
    obs_end: int,
    reg_min: int,
    reg_max: int,
    chunksize: int,
) -> pd.DataFrame:
    """Users whose account registration falls in the observation window."""
    parts: list[pd.DataFrame] = []
    for chunk in iter_csv_chunks(
        members_path,
        chunksize=chunksize,
        usecols=["msno", "registration_init_time"],
    ):
        reg = pd.to_numeric(chunk["registration_init_time"], errors="coerce")
        mask = reg.between(obs_start, obs_end) & reg.between(reg_min, reg_max)
        if mask.any():
            parts.append(
                pd.DataFrame(
                    {
                        "msno": chunk.loc[mask, "msno"].values,
                        "registration_init_time": reg[mask].astype("Int64"),
                    }
                )
            )

    if not parts:
        return pd.DataFrame(columns=["msno", "registration_init_time"])

    out = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["msno"])
    out["msno"] = out["msno"].astype(str)
    return out


def scan_first_transaction_dates(
    transaction_paths: list[Path],
    eligible_ids: set[str],
    *,
    chunksize: int,
) -> dict[str, int]:
    """Min transaction_date per msno among eligible_ids (streaming)."""
    first_txn: dict[str, int] = {}

    for path in transaction_paths:
        for chunk in iter_csv_chunks(
            path,
            chunksize=chunksize,
            usecols=["msno", "transaction_date"],
        ):
            chunk = chunk.assign(msno=chunk["msno"].astype(str))
            sub = chunk[chunk["msno"].isin(eligible_ids)]
            if sub.empty:
                continue
            txn = pd.to_numeric(sub["transaction_date"], errors="coerce")
            sub = sub.assign(transaction_date=txn).dropna(subset=["transaction_date"])
            if sub.empty:
                continue
            mins = sub.groupby("msno", as_index=False)["transaction_date"].min()
            for row in mins.itertuples(index=False):
                msno, dt = row.msno, int(row.transaction_date)
                prev = first_txn.get(msno)
                if prev is None or dt < prev:
                    first_txn[msno] = dt

    return first_txn


def build_eligible_cohort(cfg: KKBoxSubsampleConfig) -> pd.DataFrame:
    """
    Return eligible users with registration_init_time and first_transaction_date.
    Saves `eligible_users.csv` under cfg.out_dir.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    members_path = cfg.resolve_existing(cfg.members)
    if members_path is None:
        raise FileNotFoundError(f"members file not found under {cfg.raw_dir}")

    print(f"Scanning registrations: {members_path}")
    registered = scan_registered_in_observation(
        members_path,
        obs_start=cfg.obs_start,
        obs_end=cfg.obs_end,
        reg_min=cfg.reg_min,
        reg_max=cfg.reg_max,
        chunksize=cfg.chunksize,
    )
    print(f"  Registered in observation window: {len(registered):,}")

    txn_paths: list[Path] = []
    for name in (cfg.transactions, cfg.transactions_v2):
        p = cfg.resolve_existing(name)
        if p is not None:
            txn_paths.append(p)
    if not txn_paths:
        raise FileNotFoundError("No transactions file found")

    eligible_ids = set(registered["msno"])
    print(f"Scanning first transaction dates ({len(txn_paths)} file(s))...")
    first_txn = scan_first_transaction_dates(
        txn_paths, eligible_ids, chunksize=cfg.chunksize
    )
    print(f"  With at least one transaction: {len(first_txn):,}")

    cohort = registered[registered["msno"].isin(first_txn)].copy()
    cohort["first_transaction_date"] = cohort["msno"].map(first_txn)
    print(f"  With registration + transaction: {len(cohort):,}")

    cohort_path = cfg.out_dir / "eligible_users.csv"
    cohort.to_csv(cohort_path, index=False)
    print(f"Saved eligible cohort -> {cohort_path}")
    return cohort


def sample_users(
    cohort: pd.DataFrame,
    *,
    n_sample: int,
    random_state: int,
) -> pd.DataFrame:
    """Random sample of n_sample users from cohort."""
    if len(cohort) < n_sample:
        raise ValueError(
            f"Eligible cohort has {len(cohort)} users; cannot sample {n_sample}."
        )
    return cohort.sample(n=n_sample, random_state=random_state).reset_index(drop=True)


def extract_table_for_sample(
    source_path: Path,
    sample_ids: set[str],
    out_path: Path,
    *,
    chunksize: int,
) -> int:
    """Stream-filter source CSV/7z to out_path; return row count written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    header_written = False

    for chunk in iter_csv_chunks(source_path, chunksize=chunksize):
        chunk = chunk.assign(msno=chunk["msno"].astype(str))
        sub = chunk[chunk["msno"].isin(sample_ids)]
        if sub.empty:
            continue
        sub.to_csv(
            out_path,
            mode="a",
            header=not header_written,
            index=False,
        )
        header_written = True
        n_rows += len(sub)

    if not header_written:
        # write empty file with headers from first chunk
        for chunk in iter_csv_chunks(source_path, chunksize=1):
            pd.DataFrame(columns=chunk.columns).to_csv(out_path, index=False)
            break

    return n_rows


def extract_subsample_tables(
    cfg: KKBoxSubsampleConfig,
    sample_ids: set[str],
) -> dict[str, int]:
    """Write filtered members, transactions, user_logs for sample_ids."""
    subsample_dir = cfg.out_dir / "subsample"
    subsample_dir.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}

    path_map = {
        "members": cfg.resolve_existing(cfg.members),
        "transactions": cfg.resolve_existing(cfg.transactions),
        "user_logs": cfg.resolve_existing(cfg.user_logs),
    }

    for key in cfg.files_to_extract:
        src = path_map.get(key)
        if src is None:
            print(f"  Skipping {key}: file not found")
            continue
        out = subsample_dir / f"{key}.csv"
        if out.exists():
            out.unlink()
        print(f"  Extracting {key}...")
        n = extract_table_for_sample(
            src, sample_ids, out, chunksize=cfg.chunksize
        )
        row_counts[key] = n
        print(f"    -> {out} ({n:,} rows)")

    # transactions_v2 if present
    t2 = cfg.resolve_existing(cfg.transactions_v2)
    if t2 is not None and "transactions" in cfg.files_to_extract:
        out = subsample_dir / "transactions_v2.csv"
        if out.exists():
            out.unlink()
        print("  Extracting transactions_v2...")
        n = extract_table_for_sample(
            t2, sample_ids, out, chunksize=cfg.chunksize
        )
        row_counts["transactions_v2"] = n
        print(f"    -> {out} ({n:,} rows)")

    # user_logs_v2 if present
    ul2 = cfg.resolve_existing(cfg.user_logs_v2)
    if ul2 is not None and "user_logs" in cfg.files_to_extract:
        out = subsample_dir / "user_logs_v2.csv"
        if out.exists():
            out.unlink()
        print("  Extracting user_logs_v2...")
        n = extract_table_for_sample(ul2, sample_ids, out, chunksize=cfg.chunksize)
        row_counts["user_logs_v2"] = n
        print(f"    -> {out} ({n:,} rows)")

    return row_counts


TRANSACTION_MERGE_COLS = [
    "msno",
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "transaction_date",
    "membership_expire_date",
    "is_cancel",
]

USER_LOGS_SUM_COLS = [
    "num_25",
    "num_50",
    "num_75",
    "num_985",
    "num_100",
    "num_unq",
    "total_secs",
]


def merge_subsample_transactions(subsample_dir: Path) -> pd.DataFrame:
    """Concat transactions + transactions_v2, dedupe, write transactions_merged.csv."""
    parts: list[pd.DataFrame] = []
    for name in ("transactions.csv", "transactions_v2.csv"):
        path = subsample_dir / name
        if path.exists():
            parts.append(pd.read_csv(path))

    if not parts:
        raise FileNotFoundError(
            f"No transaction subsample files in {subsample_dir}"
        )

    merged = pd.concat(parts, ignore_index=True)
    merged["msno"] = merged["msno"].astype(str)
    merged = merged.drop_duplicates(subset=TRANSACTION_MERGE_COLS).sort_values(
        ["msno", "transaction_date", "membership_expire_date"],
        kind="mergesort",
    )
    out = subsample_dir / "transactions_merged.csv"
    merged.to_csv(out, index=False)
    print(f"  Merged transactions -> {out} ({len(merged):,} rows)")
    return merged


def merge_subsample_user_logs(subsample_dir: Path) -> pd.DataFrame:
    """
    Concat user_logs + user_logs_v2, aggregate to one row per (msno, date).
    Writes user_logs_daily.csv with hours = total_secs / 3600.
    """
    parts: list[pd.DataFrame] = []
    for name in ("user_logs.csv", "user_logs_v2.csv"):
        path = subsample_dir / name
        if path.exists():
            parts.append(pd.read_csv(path))

    if not parts:
        raise FileNotFoundError(f"No user_logs subsample files in {subsample_dir}")

    logs = pd.concat(parts, ignore_index=True)
    logs["msno"] = logs["msno"].astype(str)
    sum_cols = [c for c in USER_LOGS_SUM_COLS if c in logs.columns]
    daily = logs.groupby(["msno", "date"], as_index=False)[sum_cols].sum()
    daily["hours"] = daily["total_secs"] / 3600.0
    out = subsample_dir / "user_logs_daily.csv"
    daily.to_csv(out, index=False)
    print(f"  Merged user logs -> {out} ({len(daily):,} user-days)")
    return daily


def merge_subsample_tables(cfg: KKBoxSubsampleConfig) -> dict[str, int]:
    """Build single merged transaction and user-log files in subsample/."""
    subsample_dir = cfg.out_dir / "subsample"
    counts: dict[str, int] = {}

    txn_path = subsample_dir / "transactions.csv"
    txn_v2_path = subsample_dir / "transactions_v2.csv"
    if txn_path.exists() or txn_v2_path.exists():
        print("Merging transaction subsamples...")
        merged_txn = merge_subsample_transactions(subsample_dir)
        counts["transactions_merged"] = len(merged_txn)
        if not cfg.keep_partial_extracts:
            txn_path.unlink(missing_ok=True)
            txn_v2_path.unlink(missing_ok=True)

    ul_path = subsample_dir / "user_logs.csv"
    ul_v2_path = subsample_dir / "user_logs_v2.csv"
    if ul_path.exists() or ul_v2_path.exists():
        print("Merging user_log subsamples...")
        daily = merge_subsample_user_logs(subsample_dir)
        counts["user_logs_daily"] = len(daily)
        if not cfg.keep_partial_extracts:
            ul_path.unlink(missing_ok=True)
            ul_v2_path.unlink(missing_ok=True)

    return counts


def run_pipeline(cfg: KKBoxSubsampleConfig) -> pd.DataFrame:
    """
    Full pipeline: eligible cohort -> sample 1000 -> extract tables.
    Returns the sample user metadata DataFrame.
    """
    cohort = build_eligible_cohort(cfg)

    print(f"Sampling {cfg.n_sample} users...")
    sample = sample_users(
        cohort, n_sample=cfg.n_sample, random_state=cfg.random_state
    )
    sample["msno"] = sample["msno"].astype(str)

    sample_ids = set(sample["msno"].astype(str))
    sample_meta_path = cfg.out_dir / "sample_users.csv"
    sample_ids_path = cfg.out_dir / "sample_msno.csv"
    sample.to_csv(sample_meta_path, index=False)
    pd.DataFrame({"msno": sorted(sample_ids)}).to_csv(sample_ids_path, index=False)
    print(f"Saved sample metadata -> {sample_meta_path}")
    print(f"Saved sample ids -> {sample_ids_path}")

    print("Extracting subsample tables (chunked)...")
    counts = extract_subsample_tables(cfg, sample_ids)
    for name, n in counts.items():
        print(f"  {name}: {n:,} rows")

    if cfg.merge_outputs:
        merge_counts = merge_subsample_tables(cfg)
        counts.update(merge_counts)

    return sample


if __name__ == "__main__":
    # Example — edit paths and run: python project/kkbox_subsample.py
    cfg = KKBoxSubsampleConfig(
        raw_dir=Path("/path/to/kkbox/raw"),
        out_dir=Path("/path/to/kkbox/subsample"),
    )
    run_pipeline(cfg)
