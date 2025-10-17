import codecs
import csv
import io
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


CANONICAL_COLUMNS = ["date", "description", "amount", "account_name", "balance"]

DATE_CANDIDATES = [
    "date",
    "transaction date",
    "posting date",
    "posted date",
    "date posted",
    "date/time",
]

DESCRIPTION_CANDIDATES = [
    "description",
    "transaction details",
    "details",
    "memo",
    "narrative",
    "description 1",
]

AMOUNT_CANDIDATES = [
    "amount",
    "transaction amount",
    "cad$",
    "amount ($)",
    "amount (cad)",
    "cad amount",
]

DEBIT_CANDIDATES = ["debit", "withdrawal", "withdrawals", "debito"]

CREDIT_CANDIDATES = ["credit", "deposit", "deposits", "credito"]

BALANCE_CANDIDATES = ["balance", "account balance", "running balance", "available balance"]

ACCOUNT_CANDIDATES = ["account", "account name", "account type", "product"]

TRANSFER_KEYWORDS = [
    "transfer",
    "xfer",
    "between accounts",
    "online banking transfer",
    "customer transfer",
    "etfr",
]

INTERNAL_TRANSFER_DESCRIPTIONS = {"customer transfer cr.", "customer transfer dr."}


def detect_encoding(raw_bytes: bytes) -> str:
    if raw_bytes.startswith(codecs.BOM_UTF16_LE):
        return "utf-16"
    try:
        raw_bytes.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "latin-1"


def detect_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except csv.Error:
        return ","


def infer_account_name(file_name: Optional[str], account_column: Optional[pd.Series]) -> str:
    if account_column is not None and account_column.notna().any():
        value = account_column.dropna().iloc[0].strip()
        if value:
            return value.lower()
    if not file_name:
        return "chequing"
    lowered = file_name.lower()
    if "sav" in lowered:
        return "savings"
    if "visa" in lowered or "credit" in lowered or "card" in lowered:
        return "credit card"
    if "tfsa" in lowered:
        return "tfsa"
    return "chequing"


def _clean_string(value: Optional[str]) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        value = str(value)
    elif not isinstance(value, str):
        value = str(value)
    value = value.replace("\xa0", " ").strip()
    value = re.sub(r"\s+", " ", value)
    printable = "".join(ch for ch in value if ch.isprintable())
    return printable.strip()


def _clean_numeric(value: str) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    value = value.replace(",", "").replace("$", "")
    value = value.replace("(", "-").replace(")", "")
    try:
        return float(value)
    except ValueError:
        return None


def _normalize_columns(columns: Sequence[str]) -> List[str]:
    normalized = []
    for col in columns:
        col = col.strip().lower()
        col = re.sub(r"\s+", " ", col)
        normalized.append(col)
    return normalized


def _find_first_match(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    for candidate in candidates:
        for col in columns:
            if col.endswith(candidate):
                return col
    return None


def read_scotiabank_csv(raw_bytes: bytes, file_name: Optional[str] = None) -> pd.DataFrame:
    encoding = detect_encoding(raw_bytes)
    decoded = raw_bytes.decode(encoding, errors="ignore")
    sample = decoded[:4096]
    delimiter = detect_delimiter(sample)
    csv_buffer = io.StringIO(decoded)
    df = pd.read_csv(csv_buffer, delimiter=delimiter, dtype=str).dropna(how="all")
    df.columns = _normalize_columns(df.columns)

    date_column = _find_first_match(df.columns, DATE_CANDIDATES)
    description_column = _find_first_match(df.columns, DESCRIPTION_CANDIDATES)
    balance_column = _find_first_match(df.columns, BALANCE_CANDIDATES)
    account_column_name = _find_first_match(df.columns, ACCOUNT_CANDIDATES)
    amount_column = _find_first_match(df.columns, AMOUNT_CANDIDATES)

    debit_column = _find_first_match(df.columns, DEBIT_CANDIDATES)
    credit_column = _find_first_match(df.columns, CREDIT_CANDIDATES)

    if date_column is None or description_column is None:
        raise ValueError("Required columns not found in uploaded file.")

    account_series = df[account_column_name] if account_column_name else None
    account_name = infer_account_name(file_name, account_series)

    normalized = pd.DataFrame()
    normalized["date"] = pd.to_datetime(df[date_column], errors="coerce")
    normalized["description"] = df[description_column].apply(_clean_string)

    amount_values: Optional[pd.Series] = None
    if amount_column:
        amount_values = df[amount_column].apply(_clean_numeric)
    elif debit_column or credit_column:
        debit_series = df[debit_column].apply(_clean_numeric) if debit_column else None
        credit_series = df[credit_column].apply(_clean_numeric) if credit_column else None
        amount_values = pd.Series(0.0, index=df.index, dtype="float64")
        if debit_series is not None:
            amount_values = amount_values - debit_series.fillna(0.0)
        if credit_series is not None:
            amount_values = amount_values + credit_series.fillna(0.0)
    else:
        raise ValueError("Could not determine amount columns in uploaded file.")

    normalized["amount"] = pd.to_numeric(amount_values, errors="coerce")
    normalized["account_name"] = account_name

    if balance_column:
        normalized["balance"] = pd.to_numeric(
            df[balance_column].apply(_clean_numeric), errors="coerce"
        )
    else:
        normalized["balance"] = pd.NA

    normalized = normalized.dropna(subset=["date", "amount"])
    normalized = normalized.sort_values("date").reset_index(drop=True)
    return normalized


def load_and_normalize_files(file_payloads: Iterable[Tuple[str, bytes]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for file_name, content in file_payloads:
        frame = read_scotiabank_csv(content, file_name=file_name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    combined = remove_internal_transfers(combined)
    return combined


def _matches_transfer_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in TRANSFER_KEYWORDS)


def remove_internal_transfers(df: pd.DataFrame, tolerance_days: int = 2) -> pd.DataFrame:
    if df.empty:
        return df
    working = df.copy().reset_index(drop=True)
    working["description_lower"] = working["description"].str.lower()
    mask_customer = working["description_lower"].isin(
        tuple(desc.lower() for desc in INTERNAL_TRANSFER_DESCRIPTIONS)
    )
    working = working[~mask_customer]

    working["transfer_flag"] = working["description"].apply(
        lambda desc: _matches_transfer_keyword(desc if isinstance(desc, str) else "")
    )
    working["abs_amount"] = working["amount"].abs().round(2)

    positives = working[(working["amount"] > 0) & working["transfer_flag"]]
    negatives = working[(working["amount"] < 0) & working["transfer_flag"]]

    if positives.empty or negatives.empty:
        cleaned = working.drop(columns=["description_lower", "transfer_flag", "abs_amount"])
        return cleaned.reset_index(drop=True)

    positives = positives.assign(pair_index=positives.index)
    negatives = negatives.assign(pair_index=negatives.index)

    potential_pairs = positives.merge(
        negatives,
        on="abs_amount",
        suffixes=("_pos", "_neg"),
    )
    if potential_pairs.empty:
        cleaned = working.drop(columns=["description_lower", "transfer_flag", "abs_amount"])
        return cleaned.reset_index(drop=True)

    potential_pairs["date_diff"] = (potential_pairs["date_pos"] - potential_pairs["date_neg"]).abs()
    potential_pairs = potential_pairs[
        potential_pairs["date_diff"] <= pd.Timedelta(days=tolerance_days)
    ]
    potential_pairs = potential_pairs.sort_values("date_diff")

    to_drop = set()
    used_pos = set()
    used_neg = set()

    for _, row in potential_pairs.iterrows():
        pos_idx = int(row["pair_index_pos"])
        neg_idx = int(row["pair_index_neg"])
        if pos_idx in used_pos or neg_idx in used_neg:
            continue
        used_pos.add(pos_idx)
        used_neg.add(neg_idx)
        to_drop.add(pos_idx)
        to_drop.add(neg_idx)

    working = working.drop(index=list(to_drop))
    working = working.drop(columns=["description_lower", "transfer_flag", "abs_amount"])
    return working.reset_index(drop=True)


def serialize_uploaded_files(files: Sequence) -> Tuple[Tuple[str, bytes], ...]:
    serialized: List[Tuple[str, bytes]] = []
    for file in files:
        name = getattr(file, "name", "upload.csv")
        content = file.getvalue()
        serialized.append((name, content))
    return tuple(serialized)


def snapshot_file_list(files: Sequence) -> str:
    names = [getattr(file, "name", "upload.csv") for file in files]
    return ", ".join(names)
