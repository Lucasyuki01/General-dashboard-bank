import difflib
import json
import os
import re
from functools import lru_cache
from typing import Dict, Optional, Tuple
from urllib import error, request

import numpy as np
import pandas as pd


DEFAULT_MAPPING = {
    "Earnings": ("Income", "General"),
    "Expenses": ("Others", "Others"),
}

DETERMINISTIC_KEYWORDS: Dict[str, Tuple[str, str]] = {
    "payroll": ("Income", "Salary"),
    "salary": ("Income", "Salary"),
    "direct deposit": ("Income", "Salary"),
    "interest": ("Income", "Interest"),
    "dividend": ("Income", "Investments"),
    "gst credit": ("Income", "Government"),
    "rent": ("Housing", "Rent"),
    "mortgage": ("Housing", "Mortgage"),
    "hydro": ("Utilities", "Electricity"),
    "enbridge": ("Utilities", "Gas"),
    "rogers": ("Utilities", "Telecom"),
    "bell": ("Utilities", "Telecom"),
    "telus": ("Utilities", "Telecom"),
    "shaw": ("Utilities", "Telecom"),
    "internet": ("Utilities", "Telecom"),
    "insurance": ("Insurance", "Premiums"),
    "property tax": ("Housing", "Taxes"),
    "scotia visa payment": ("Transfers", "Credit Card Payment"),
    "visa payment": ("Transfers", "Credit Card Payment"),
    "xfer": ("Transfers", "Internal Transfer"),
    "transfer": ("Transfers", "Internal Transfer"),
    "etfr": ("Transfers", "Internal Transfer"),
    "uber": ("Transport", "Ride Share"),
    "lyft": ("Transport", "Ride Share"),
    "shell": ("Transport", "Fuel"),
    "esso": ("Transport", "Fuel"),
    "petro": ("Transport", "Fuel"),
    "petro-canada": ("Transport", "Fuel"),
    "costco": ("Living", "Groceries"),
    "loblaws": ("Living", "Groceries"),
    "walmart": ("Living", "Groceries"),
    "sobeys": ("Living", "Groceries"),
    "no frills": ("Living", "Groceries"),
    "shoppers": ("Health", "Pharmacy"),
    "starbucks": ("Food & Drink", "Coffee"),
    "tim hortons": ("Food & Drink", "Coffee"),
    "mcdonald": ("Food & Drink", "Fast Food"),
    "netflix": ("Entertainment", "Streaming"),
    "spotify": ("Entertainment", "Streaming"),
    "canadian tire": ("Living", "Home"),
    "home depot": ("Living", "Home Improvement"),
    "amazon": ("Shopping", "Online Retail"),
    "airbnb": ("Travel", "Accommodation"),
    "uber eats": ("Food & Drink", "Delivery"),
}

FUZZY_TARGETS: Dict[str, Tuple[str, str]] = {
    "starbuck": ("Food & Drink", "Coffee"),
    "mcdonalds": ("Food & Drink", "Fast Food"),
    "wendys": ("Food & Drink", "Fast Food"),
    "subway": ("Food & Drink", "Fast Food"),
    "shoppers drug mart": ("Health", "Pharmacy"),
    "ikea": ("Living", "Home"),
    "canadian tire": ("Living", "Home"),
    "costco wholesale": ("Living", "Groceries"),
    "longos": ("Living", "Groceries"),
    "freshco": ("Living", "Groceries"),
    "no frills": ("Living", "Groceries"),
    "uber trip": ("Transport", "Ride Share"),
    "uber eats": ("Food & Drink", "Delivery"),
}

API_URL = os.getenv("PF_CLASSIFY_API_URL")
API_KEY = os.getenv("PF_CLASSIFY_API_KEY")
API_TIMEOUT = 3


def normalize(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def deterministic_lookup(description: str) -> Optional[Tuple[str, str]]:
    for keyword, mapping in DETERMINISTIC_KEYWORDS.items():
        if keyword in description:
            return mapping
    return None


def fuzzy_lookup(description: str) -> Optional[Tuple[str, str]]:
    tokens = [token for token in re.split(r"[^\w&]+", description) if token]
    candidates = list(FUZZY_TARGETS.keys())
    for token in tokens:
        matches = difflib.get_close_matches(token, candidates, n=1, cutoff=0.92)
        if matches:
            return FUZZY_TARGETS[matches[0]]
    matches = difflib.get_close_matches(description, candidates, n=1, cutoff=0.8)
    if matches:
        return FUZZY_TARGETS[matches[0]]
    return None


@lru_cache(maxsize=128)
def _call_enrichment_api(description: str) -> Optional[Tuple[str, str]]:
    if not API_URL or not API_KEY:
        return None
    payload = json.dumps({"description": description}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    req = request.Request(API_URL, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=API_TIMEOUT) as response:
            if response.status >= 400:
                return None
            data = json.loads(response.read().decode("utf-8"))
            category = data.get("category")
            sub_category = data.get("sub_category")
            if category and sub_category:
                return category, sub_category
            if category:
                return category, "Others"
            return None
    except (error.URLError, ValueError, TimeoutError):
        return None


def classify_transactions(df: pd.DataFrame, use_api: bool = False) -> pd.DataFrame:
    if df.empty:
        empty = df.copy()
        if "category" not in empty.columns:
            empty["category"] = pd.Series(dtype="object")
        if "sub_category" not in empty.columns:
            empty["sub_category"] = pd.Series(dtype="object")
        return empty

    classified = df.copy()
    if "class" not in classified.columns:
        classified["class"] = np.where(classified["amount"] > 0, "Earnings", "Expenses")

    categories = []
    sub_categories = []

    for description, row_class in zip(classified["description"], classified["class"]):
        normalized_description = normalize(str(description))
        mapping = deterministic_lookup(normalized_description)
        if mapping is None:
            mapping = fuzzy_lookup(normalized_description)
        if mapping is None and use_api:
            api_mapping = _call_enrichment_api(normalized_description)
            if api_mapping:
                mapping = api_mapping
        if mapping is None:
            fallback = DEFAULT_MAPPING.get(row_class, DEFAULT_MAPPING["Expenses"])
            mapping = fallback
        categories.append(mapping[0])
        sub_categories.append(mapping[1])

    classified["category"] = categories
    classified["sub_category"] = sub_categories
    return classified
