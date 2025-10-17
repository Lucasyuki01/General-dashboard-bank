import datetime as dt
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from charts import category_pie, monthly_amount_bar, net_worth_line, weekday_average_bar
from cleaning import clean_transactions
from classify import classify_transactions
from io_utils import (
    load_and_normalize_files,
    read_scotiabank_csv,
    serialize_uploaded_files,
    snapshot_file_list,
)


st.set_page_config(page_title="Personal Finance Dashboard", layout="wide")


SAMPLE_DATA = """Date,Description,Debit,Credit,Balance
2024-01-02,Payroll Deposit,,2800.00,4200.55
2024-01-03,Tim Hortons,4.58,,4195.97
2024-01-04,Rent Downtown Apt,1800.00,,2395.97
2024-01-05,Uber Trip 1234,18.25,,2377.72
2024-01-05,Scotia Savings Transfer,,300.00,2677.72
2024-01-05,Scotia Savings Transfer,300.00,,2377.72
2024-01-06,Groceries - Loblaws,126.38,,2251.34
2024-01-07,Netflix.com,16.99,,2234.35
2024-01-08,Costco Wholesale,210.45,,2023.90
2024-01-10,Shell Canada,62.44,,1961.46
2024-01-15,Insurance Premium,95.12,,1866.34
2024-01-20,Interest Payment,,4.22,1870.56
2024-01-23,Rogers Communications,89.50,,1781.06
2024-01-25,Starbucks Coffee,8.76,,1772.30
2024-01-28,Side Gig Payment,,420.00,2192.30
2024-02-02,Payroll Deposit,,2800.00,4992.30
2024-02-03,Groceries - No Frills,135.12,,4857.18
2024-02-04,Uber Eats,32.55,,4824.63
2024-02-05,Mortgage Payment,1600.00,,3224.63
2024-02-07,Hydro One,82.30,,3142.33
2024-02-09,Spotify P07,11.99,,3130.34
2024-02-10,Costco Wholesale,195.75,,2934.59
2024-02-12,Interest Payment,,4.45,2939.04
2024-02-15,Scotia Visa Payment,450.00,,2489.04
2024-02-18,Shoppers Drug Mart,47.33,,2441.71
2024-02-20,Airbnb Booking,320.00,,2121.71
2024-02-22,Amazon Marketplace,68.90,,2052.81
2024-02-25,Side Gig Payment,,460.00,2512.81
"""


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    return read_scotiabank_csv(SAMPLE_DATA.encode("utf-8"), file_name="sample.csv")


@st.cache_data(show_spinner=False)
def ingest_uploaded_files(serialized_files: Tuple[Tuple[str, bytes], ...]) -> pd.DataFrame:
    return load_and_normalize_files(serialized_files)


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: df.to_csv(index=False)})
def transform_dataset(raw_df: pd.DataFrame, use_api: bool) -> pd.DataFrame:
    cleaned = clean_transactions(raw_df)
    classified = classify_transactions(cleaned, use_api=use_api)
    return classified


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def ensure_state_defaults(min_date: pd.Timestamp, max_date: pd.Timestamp) -> None:
    start_dt = dt.datetime.combine(min_date.date(), dt.time.min)
    end_dt = dt.datetime.combine(max_date.date(), dt.time.max)
    if "date_picker" not in st.session_state:
        st.session_state["date_picker"] = (min_date.date(), max_date.date())
    if "date_slider" not in st.session_state:
        st.session_state["date_slider"] = (start_dt, end_dt)


def set_full_period(min_date: pd.Timestamp, max_date: pd.Timestamp) -> None:
    st.session_state["date_picker"] = (min_date.date(), max_date.date())
    st.session_state["date_slider"] = (
        dt.datetime.combine(min_date.date(), dt.time.min),
        dt.datetime.combine(max_date.date(), dt.time.max),
    )


def on_date_picker_change() -> None:
    start_date, end_date = st.session_state["date_picker"]
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    new_slider = (
        dt.datetime.combine(start_date, dt.time.min),
        dt.datetime.combine(end_date, dt.time.max),
    )
    if st.session_state.get("date_slider") != new_slider:
        st.session_state["date_slider"] = new_slider


def on_slider_change() -> None:
    start_dt, end_dt = st.session_state["date_slider"]
    start_date = start_dt.date()
    end_date = end_dt.date()
    new_picker = (start_date, end_date)
    if st.session_state.get("date_picker") != new_picker:
        st.session_state["date_picker"] = new_picker


def hero_section() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    st.title("Personal Finance Dashboard")
    source_label = st.session_state.get("data_source", "No data loaded")
    st.caption(f"Data source: {source_label}")

    st.markdown("### Welcome")
    st.write(
        "Upload one or more Scotiabank CSV exports to explore your finances, or get started "
        "instantly with a curated sample dataset."
    )

    uploaded_files = st.file_uploader(
        "Drag and drop one or more CSV files",
        accept_multiple_files=True,
        type=["csv"],
    )
    col1, col2 = st.columns([3, 1])
    with col2:
        sample_clicked = st.button("Use sample data", use_container_width=True)
    if sample_clicked:
        sample_df = load_sample_data()
        st.session_state["raw_df"] = sample_df
        st.session_state["data_source"] = "Sample dataset"
        st.session_state["data_signature"] = None
        return sample_df, "Sample dataset"

    if uploaded_files:
        serialized = serialize_uploaded_files(uploaded_files)
        uploaded_df = ingest_uploaded_files(serialized)
        label = f"Uploaded files: {snapshot_file_list(uploaded_files)}"
        st.session_state["raw_df"] = uploaded_df
        st.session_state["data_source"] = label
        st.session_state["data_signature"] = None
        return uploaded_df, label

    raw_df = st.session_state.get("raw_df")
    label = st.session_state.get("data_source")
    return raw_df, label


def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    start_dt, end_dt = st.session_state["date_slider"]
    filtered = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

    selected_class = st.session_state.get("filter_class")
    if selected_class:
        filtered = filtered[filtered["class"].isin(selected_class)]

    selected_category = st.session_state.get("filter_category")
    if selected_category:
        filtered = filtered[filtered["category"].isin(selected_category)]

    selected_sub_category = st.session_state.get("filter_sub_category")
    if selected_sub_category:
        filtered = filtered[filtered["sub_category"].isin(selected_sub_category)]

    selected_account = st.session_state.get("filter_account")
    if selected_account:
        filtered = filtered[filtered["account_name"].isin(selected_account)]
    return filtered


def render_filter_widgets(df: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp) -> None:
    with st.sidebar:
        ensure_state_defaults(min_date, max_date)

        st.button(
            "Select full period",
            on_click=set_full_period,
            args=(min_date, max_date),
            use_container_width=True,
        )
        st.date_input(
            "Date range",
            value=st.session_state["date_picker"],
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="date_picker",
            on_change=on_date_picker_change,
        )
        st.slider(
            "Timeline",
            min_value=dt.datetime.combine(min_date.date(), dt.time.min),
            max_value=dt.datetime.combine(max_date.date(), dt.time.max),
            value=st.session_state["date_slider"],
            key="date_slider",
            on_change=on_slider_change,
        )

        class_options = sorted(df["class"].dropna().unique().tolist())
        st.session_state["filter_class"] = st.multiselect(
            "Class",
            options=class_options,
            default=class_options,
        )

        category_options = sorted(df["category"].dropna().unique().tolist())
        st.session_state["filter_category"] = st.multiselect(
            "Category",
            options=category_options,
            default=category_options,
        )

        sub_options = sorted(df["sub_category"].dropna().unique().tolist())
        st.session_state["filter_sub_category"] = st.multiselect(
            "Sub-category",
            options=sub_options,
            default=sub_options,
        )

        account_options = sorted(df["account_name"].dropna().unique().tolist())
        st.session_state["filter_account"] = st.multiselect(
            "Account",
            options=account_options,
            default=account_options,
        )


def kpi_section(filtered: pd.DataFrame) -> None:
    earnings = filtered[filtered["class"] == "Earnings"]["amount"].sum()
    expenses = filtered[filtered["class"] == "Expenses"]["amount"].sum()
    delta = earnings + expenses
    mean_purchase = (
        filtered[filtered["class"] == "Expenses"]["amount"].apply(abs).mean()
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Earnings", format_currency(earnings))
    col2.metric("Total Expenses", format_currency(abs(expenses)))
    col3.metric("Delta", format_currency(delta))
    col4.metric(
        "Average Purchase Amount",
        format_currency(mean_purchase if not pd.isna(mean_purchase) else 0.0),
    )


def expenses_section(expenses_df: pd.DataFrame) -> None:
    st.subheader("Expenses")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(monthly_amount_bar(expenses_df, "Expenses", "Monthly Expenses"), use_container_width=True)
    with col2:
        st.plotly_chart(category_pie(expenses_df, "Expenses", "Expenses by Category"), use_container_width=True)

    st.plotly_chart(weekday_average_bar(expenses_df, "Average Expense by Weekday"), use_container_width=True)

    with st.expander("Expense transactions", expanded=False):
        display_df = expenses_df.copy()
        display_df["amount"] = display_df["amount"].apply(lambda x: -abs(x))
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def earnings_section(earnings_df: pd.DataFrame) -> None:
    st.subheader("Earnings")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(monthly_amount_bar(earnings_df, "Earnings", "Monthly Earnings"), use_container_width=True)
    with col2:
        st.plotly_chart(category_pie(earnings_df, "Earnings", "Earnings by Category"), use_container_width=True)

    with st.expander("Earning transactions", expanded=False):
        st.dataframe(earnings_df, use_container_width=True, hide_index=True)


def net_worth_section(df: pd.DataFrame) -> None:
    st.subheader("Net Worth")
    chart = net_worth_line(df, "Net Worth Over Time")
    st.plotly_chart(chart, use_container_width=True)
    if "balance" in df.columns and df["balance"].notna().any():
        st.caption("Net worth uses account balances when provided by the source files.")
    else:
        st.caption(
            "Net worth approximated using the cumulative sum of transactions. "
            "Consider adding balance data for higher accuracy."
        )


def top_expenses_section(expenses_df: pd.DataFrame) -> None:
    st.subheader("Top 10 Expenses")
    top_expenses = expenses_df.copy()
    top_expenses["abs_amount"] = top_expenses["amount"].abs()
    top_expenses = top_expenses.sort_values("abs_amount", ascending=False).head(10)
    top_expenses["amount"] = top_expenses["amount"].apply(lambda x: -abs(x))
    with st.expander("Largest expenses this period", expanded=False):
        st.dataframe(
            top_expenses.drop(columns=["abs_amount"]),
            use_container_width=True,
            hide_index=True,
        )


def category_drilldown_section(filtered: pd.DataFrame) -> None:
    st.subheader("Category Drilldown")
    categories = sorted(filtered["category"].dropna().unique().tolist())
    if not categories:
        st.info("No categories available for drilldown.")
        return
    selected = st.selectbox("Choose a category", options=categories)
    slice_df = filtered[filtered["category"] == selected]
    if slice_df.empty:
        st.info("No records for this category in the current filter selection.")
        return
    st.plotly_chart(
        category_pie(slice_df, slice_df.iloc[0]["class"], "Sub-category distribution"),
        use_container_width=True,
    )

    class_label = slice_df.iloc[0]["class"]
    total_value = slice_df["amount"].sum()
    if class_label == "Expenses":
        total_display = abs(total_value)
        average_display = slice_df["amount"].apply(abs).mean()
    else:
        total_display = total_value
        average_display = slice_df["amount"].mean()

    col1, col2 = st.columns(2)
    col1.metric("Total", format_currency(total_display))
    col2.metric("Average per transaction", format_currency(average_display))

    with st.expander(f"Transactions in {selected}", expanded=False):
        st.dataframe(slice_df, use_container_width=True, hide_index=True)


def others_section(filtered: pd.DataFrame) -> None:
    others_df = filtered[
        (filtered["category"] == "Others") | (filtered["sub_category"] == "Others")
    ]
    with st.expander("Transactions classified as Others", expanded=False):
        if others_df.empty:
            st.write("No transactions are currently classified as Others.")
        else:
            st.dataframe(others_df, use_container_width=True, hide_index=True)


def _run_acceptance_tests() -> None:
    sample_raw = load_sample_data()
    assert not sample_raw.empty, "Sample data should contain rows."
    transformed = transform_dataset(sample_raw, use_api=False)
    required_columns = {"date", "description", "amount", "class", "category", "sub_category"}
    missing = required_columns.difference(transformed.columns)
    assert not missing, f"Missing columns after transformation: {missing}"
    allowed_classes = {"Earnings", "Expenses"}
    assert set(transformed["class"].unique()).issubset(allowed_classes), "Unexpected class labels detected."


def main() -> None:
    if not st.session_state.get("_acceptance_tests_run", False):
        try:
            _run_acceptance_tests()
        except AssertionError as exc:
            st.warning(f"Acceptance check failed: {exc}")
        st.session_state["_acceptance_tests_run"] = True

    raw_df, _ = hero_section()
    if raw_df is None or raw_df.empty:
        st.info("Load data above to unlock your personal finance insights.")
        return

    if "use_api_toggle" not in st.session_state:
        st.session_state["use_api_toggle"] = False

    with st.sidebar:
        st.header("Filters")
        use_api = st.toggle(
            "Use API enrichment",
            value=st.session_state.get("use_api_toggle", False),
            key="use_api_toggle",
            help="Call an enrichment API when local rules fail.",
        )

    cleaned = transform_dataset(raw_df, use_api=use_api)

    min_date = cleaned["date"].min()
    max_date = cleaned["date"].max()
    ensure_state_defaults(min_date, max_date)

    current_signature = st.session_state.get("data_signature")
    new_signature = (min_date, max_date, cleaned.shape[0])
    if current_signature != new_signature:
        set_full_period(min_date, max_date)
        st.session_state["data_signature"] = new_signature

    render_filter_widgets(cleaned, min_date, max_date)

    filtered = filter_dataset(cleaned)
    if filtered.empty:
        st.warning("No transactions match the current filters.")
        return

    kpi_section(filtered)
    st.divider()

    expenses_df = filtered[filtered["class"] == "Expenses"]
    earnings_df = filtered[filtered["class"] == "Earnings"]

    if not expenses_df.empty:
        expenses_section(expenses_df)
        st.divider()

    if not earnings_df.empty:
        earnings_section(earnings_df)
        st.divider()

    net_worth_section(filtered)
    st.divider()

    if not expenses_df.empty:
        top_expenses_section(expenses_df)
        st.divider()

    category_drilldown_section(filtered)
    st.divider()

    others_section(filtered)


if __name__ == "__main__":
    main()
