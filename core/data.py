# core/data.py
import numpy as np
import pandas as pd

WEEK_ORDER = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]


def coerce_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float)):
        return x == 1
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "booked", "sold", "reserved"}
    return False


def ensure_datetime_col(df: pd.DataFrame) -> pd.DataFrame:
    """Find a datetime column or construct one."""
    for c in ["tee_time", "datetime", "start_time", "time", "date_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            if df[c].notna().any():
                df["tee_time"] = df[c]
                return df

    # Try (date + time)
    date_cols = [c for c in df.columns if "date" in c.lower()]
    time_cols = [c for c in df.columns if "time" in c.lower()]

    if date_cols and time_cols:
        df["tee_time"] = pd.to_datetime(
            df[date_cols[0]].astype(str) + " " + df[time_cols[0]].astype(str),
            errors="coerce",
        )
        if df["tee_time"].notna().any():
            return df

    raise ValueError("No datetime column found. Provide a tee_time or (date + time).")


def clean_teetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize tee-sheet CSV into a standard format."""
    df = df.copy()
    df = ensure_datetime_col(df)

    # Normalize price
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price"] = np.nan

    # Booked flag
    book_col = next(
        (c for c in df.columns if c.lower() in {
            "booked", "is_booked", "reserved", "filled", "status"
        }),
        None,
    )
    if book_col:
        df["booked"] = df[book_col].apply(coerce_bool)
    else:
        df["booked"] = False

    # Add time fields
    df["weekday"] = pd.Categorical(
        df["tee_time"].dt.day_name(), categories=WEEK_ORDER, ordered=True
    )
    df["hour"] = df["tee_time"].dt.hour
    df["date"] = df["tee_time"].dt.date

    # Price cleaning
    if df["price"].isna().any():
        group_med = df.groupby(["weekday", "hour"])["price"].transform("median")
        df["price"] = df["price"].fillna(group_med).fillna(df["price"].median())

    return df.sort_values("tee_time").reset_index(drop=True)


def add_time_bins(df: pd.DataFrame, slot_minutes: int = 10) -> pd.DataFrame:
    """Create slot_index / slot_label fields based on slot interval."""
    df = df.copy()
    dt = df["tee_time"]

    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    slot_index = (minute_of_day // slot_minutes).astype(int)
    slot_start_min = slot_index * slot_minutes

    slot_hour = (slot_start_min // 60).astype(int)
    slot_min = (slot_start_min % 60).astype(int)

    df["slot_index"] = slot_index
    df["slot_minutes"] = slot_minutes
    df["slot_label"] = slot_hour.astype(str).str.zfill(2) + ":" + slot_min.astype(str).str.zfill(2)
    df["slot_time"] = pd.to_datetime(
        df["tee_time"].dt.date.astype(str) + " " + df["slot_label"],
        errors="coerce"
    )
    df["slot_hour"] = slot_hour
    df["slot_minute"] = slot_min

    return df


def fmt_time_ampm(h: int, m: int) -> str:
    hh = h % 12 or 12
    ampm = "AM" if h < 12 else "PM"
    return f"{hh}:{m:02d}{ampm}"


def kpis(df: pd.DataFrame):
    total = len(df)
    booked = int(df["booked"].sum())
    util = booked / total if total else 0.0
    revenue = float(df.loc[df["booked"], "price"].sum())
    potential = float(df["price"].sum())
    return total, booked, util, revenue, potential


def daily_utilization(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("date").agg(util=("booked", "mean")).reset_index()


def utilization_matrix_hour(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby(["weekday", "hour"]).agg(
        slots=("booked", "size"),
        booked=("booked", "sum"),
    )
    grp["util"] = np.where(
        grp["slots"] > 0, grp["booked"] / grp["slots"], np.nan
    )
    return grp["util"].unstack("hour").sort_index()
