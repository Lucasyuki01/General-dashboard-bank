import re

import numpy as np
import pandas as pd


CLASS_CATEGORIES = pd.CategoricalDtype(categories=["Earnings", "Expenses"], ordered=True)


DESCRIPTION_TRAILING_PATTERNS = [
    r"\bref\.?\s?#?\d+$",
    r"\btransaction\s?#?\d+$",
    r"\b\d{4,}$",
]


def standardize_description(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(". ")
    for pattern in DESCRIPTION_TRAILING_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date", "amount"])
    cleaned["description"] = cleaned["description"].fillna("").apply(standardize_description)
    cleaned["account_name"] = cleaned["account_name"].fillna("chequing").str.lower()
    if "balance" in cleaned.columns:
        cleaned["balance"] = pd.to_numeric(cleaned["balance"], errors="coerce")
    cleaned["class"] = np.where(cleaned["amount"] > 0, "Earnings", "Expenses")
    cleaned["class"] = cleaned["class"].astype(CLASS_CATEGORIES)
    cleaned["weekday_name"] = cleaned["date"].dt.day_name()
    cleaned["month"] = cleaned["date"].dt.to_period("M").astype(str)
    return cleaned.reset_index(drop=True)
