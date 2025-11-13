# core/pricing.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from core.data import add_time_bins, fmt_time_ampm


def featurize(tee_df: pd.DataFrame, slot_minutes: int = 10):
    df = add_time_bins(tee_df, slot_minutes=slot_minutes).copy()
    df["is_weekend"] = df["tee_time"].dt.weekday >= 5

    minute_of_day = df["slot_hour"] * 60 + df["slot_minute"]

    X = pd.DataFrame({
        "slot_index": df["slot_index"],
        "minute_of_day": minute_of_day,
        "is_weekend": df["is_weekend"].astype(int),
        "price": df["price"],
    })

    y = df["booked"].astype(int)
    meta = df[
        ["weekday", "slot_index", "slot_label", "slot_hour", "slot_minute", "price"]
    ]

    return X, y, meta


def train_model(tee_df: pd.DataFrame, slot_minutes: int = 10):
    X, y, _ = featurize(tee_df, slot_minutes=slot_minutes)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X, y)
    return clf


def expected_utilization(clf, tee_df, slot_minutes: int = 10):
    X, _, meta = featurize(tee_df, slot_minutes=slot_minutes)
    proba = clf.predict_proba(X)[:, 1]

    meta = meta.copy()
    meta["p_book"] = proba

    agg = meta.groupby(
        ["weekday", "slot_index", "slot_label", "slot_hour", "slot_minute"]
    ).agg(
        expected_util=("p_book", "mean"),
        avg_price=("price", "mean"),
    ).reset_index()

    return agg


def compute_pricing_actions(
    tee_df: pd.DataFrame,
    slot_minutes: int = 10,
    target_util: float = 0.75,
    top_n: int = 10,
):
    if tee_df.empty:
        return pd.DataFrame()

    try:
        clf = train_model(tee_df, slot_minutes=slot_minutes)
        util_df = expected_utilization(clf, tee_df, slot_minutes=slot_minutes)
    except Exception:
        tmp = add_time_bins(tee_df, slot_minutes=slot_minutes)
        grp = tmp.groupby(
            ["weekday", "slot_index", "slot_label", "slot_hour", "slot_minute"]
        ).agg(
            slots=("booked", "size"),
            booked=("booked", "sum"),
            avg_price=("price", "mean"),
        ).reset_index()

        grp["expected_util"] = np.where(
            grp["slots"] > 0, grp["booked"] / grp["slots"], 0
        )
        util_df = grp

    util_df = util_df.sort_values(
        ["expected_util", "weekday", "slot_index"]
    ).reset_index(drop=True)

    soft = util_df.head(top_n).copy()

    gap = (target_util - soft["expected_util"]).clip(lower=0)
    soft["suggested_discount"] = (0.10 + 0.20 * (gap / target_util)).clip(upper=0.35)
    soft["new_price"] = soft["avg_price"] * (1 - soft["suggested_discount"])

    soft["Time"] = soft.apply(
        lambda r: fmt_time_ampm(int(r["slot_hour"]), int(r["slot_minute"])), axis=1
    )

    soft["Expected Utilization"] = (soft["expected_util"] * 100).map(
        lambda x: f"{x:.2f}%"
    )
    soft["Average Price"] = soft["avg_price"].map(lambda x: f"${x:.2f}")
    soft["Suggested Discount"] = (
        soft["suggested_discount"] * 100
    ).map(lambda x: f"{x:.2f}%")
    soft["New Price"] = soft["new_price"].map(lambda x: f"${x:.2f}")

    pretty = soft[
        [
            "weekday",
            "Time",
            "Expected Utilization",
            "Average Price",
            "Suggested Discount",
            "New Price",
        ]
    ].rename(columns={"weekday": "Weekday"})

    return pretty

