from typing import List

import pandas as pd
import plotly.express as px


CLASS_COLOR_MAP = {
    "Earnings": "#4C78A8",
    "Expenses": "#F58518",
}

WEEKDAY_ORDER: List[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _value_for_class(df: pd.DataFrame, class_label: str) -> pd.Series:
    values = df["amount"].astype(float)
    if class_label == "Expenses":
        return -values
    return values


def monthly_amount_bar(df: pd.DataFrame, class_label: str, title: str) -> px.bar:
    if df.empty:
        return px.bar(title=title)
    data = df.copy()
    data["value"] = _value_for_class(data, class_label)
    monthly = (
        data.groupby("month", as_index=False)["value"].sum().sort_values("month")
    )
    color = CLASS_COLOR_MAP.get(class_label, "#4C78A8")
    fig = px.bar(
        monthly,
        x="month",
        y="value",
        title=title,
        labels={"month": "Month", "value": "Amount ($)"},
        color_discrete_sequence=[color],
    )
    fig.update_layout(xaxis=dict(type="category"))
    return fig


def category_pie(df: pd.DataFrame, class_label: str, title: str) -> px.pie:
    if df.empty:
        return px.pie(title=title)
    data = df.copy()
    data["value"] = _value_for_class(data, class_label)
    grouped = (
        data.groupby("category", dropna=False)["value"].sum().reset_index()
    )
    grouped = grouped[grouped["value"] != 0]
    fig = px.pie(
        grouped,
        names="category",
        values="value",
        title=title,
        color_discrete_sequence=px.colors.sequential.Blues
        if class_label == "Earnings"
        else px.colors.sequential.Oranges,
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(legend_title="Category")
    return fig


def weekday_average_bar(df: pd.DataFrame, title: str) -> px.bar:
    if df.empty:
        return px.bar(title=title)
    ordered = pd.Categorical(df["weekday_name"], categories=WEEKDAY_ORDER, ordered=True)
    data = df.copy()
    data["weekday_name"] = ordered
    data = data.dropna(subset=["weekday_name"])
    data["value"] = -data["amount"].astype(float)
    averages = (
        data.groupby("weekday_name", as_index=False)["value"].mean().sort_values("weekday_name")
    )
    fig = px.bar(
        averages,
        x="weekday_name",
        y="value",
        title=title,
        labels={"weekday_name": "Weekday", "value": "Average Amount ($)"},
        color_discrete_sequence=[CLASS_COLOR_MAP["Expenses"]],
    )
    return fig


def net_worth_line(df: pd.DataFrame, title: str) -> px.line:
    if df.empty:
        return px.line(title=title)
    data = df.copy().sort_values("date")
    if "balance" in data.columns and data["balance"].notna().any():
        networth = (
            data.dropna(subset=["balance"])
            .groupby("date", as_index=False)["balance"]
            .last()
            .rename(columns={"balance": "net_worth"})
        )
    else:
        cumulative = data.groupby("date", as_index=False)["amount"].sum()
        cumulative["net_worth"] = cumulative["amount"].cumsum()
        networth = cumulative[["date", "net_worth"]]
    fig = px.line(
        networth,
        x="date",
        y="net_worth",
        title=title,
        labels={"date": "Date", "net_worth": "Net Worth ($)"},
        color_discrete_sequence=[CLASS_COLOR_MAP["Earnings"]],
    )
    fig.update_traces(mode="lines+markers")
    return fig
