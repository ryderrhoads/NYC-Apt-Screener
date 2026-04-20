#!/usr/bin/env python3
"""
Merged first-pass OLS regression on RentHop listings.

Combines:
- v2: robust cleaning, drop logging, scraper bug fixes
- v1: better feature engineering, usability, transformations

Target: log(price_usd)
"""

from __future__ import annotations
import argparse, re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ------------------------
# CONSTANTS
# ------------------------
NYC_CENTER_LAT, NYC_CENTER_LON = 40.7580, -73.9855

BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
BASELINE_BOROUGH = "Manhattan"

PRICE_RANGE_RE = re.compile(r"\$[\d,]+\s*-\s*\$[\d,]+")
COMMERCIAL_RE = re.compile(r"\b(retail|commercial|office|storefront|garage|parking)\b", re.I)
ROOM_SHARE_RE = re.compile(r"\b(room|roommate|shared|sublet)\b", re.I)

PRICE_MIN, PRICE_MAX = 1000, 20000
BEDS_MAX, BATHS_MAX = 6, 5
SQFT_MIN, SQFT_MAX = 150, 4000
PSQFT_MAX = 15.0

# ------------------------
# HELPERS
# ------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def extract_borough(neigh):
    if not isinstance(neigh, str): return None
    last = neigh.split(",")[-1].strip()
    return last if last in BOROUGHS else None

def fix_price(row):
    raw = row.get("price_raw")
    if isinstance(raw, str) and PRICE_RANGE_RE.search(raw):
        low = raw.split("-")[0]
        digits = re.sub(r"[^\d]", "", low)
        return float(digits) if digits else None
    return float(row.get("price_usd")) if row.get("price_usd") else None

# ------------------------
# CLEANER (v2 core)
# ------------------------
class Cleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.steps = []
        self._log("start", len(df))

    def _log(self, label, after):
        before = self.steps[-1][2] if self.steps else len(self.df)
        self.steps.append((label, before-after, after))

    def keep(self, mask, label):
        self.df = self.df[mask].copy()
        self._log(label, len(self.df))
        return self

    def print(self):
        print("\n[clean] drop log:")
        for label, dropped, kept in self.steps:
            print(f"{label:<40} dropped={dropped:>7} kept={kept:>7}")

# ------------------------
# LOAD + CLEAN
# ------------------------
def load_and_clean(path: Path):
    df = pd.read_json(path, lines=True)
    df["price_usd"] = df.apply(fix_price, axis=1)

    c = Cleaner(df)

    c.keep(c.df["price_usd"].notna(), "has price")
    c.keep(c.df[["latitude","longitude","neighborhoods"]].notna().all(axis=1), "has geo")

    c.keep(
        c.df["latitude"].between(40.4,41.0) &
        c.df["longitude"].between(-74.3,-73.65),
        "NYC bbox"
    )

    c.keep(~c.df["title"].fillna("").str.contains(COMMERCIAL_RE), "not commercial")
    c.keep(~c.df["title"].fillna("").str.contains(ROOM_SHARE_RE), "not room-share")

    c.keep(c.df["bedrooms"] <= BEDS_MAX, "beds sane")
    c.keep(c.df["bathrooms"].between(0.5,BATHS_MAX), "baths sane")
    c.keep(c.df["price_usd"].between(PRICE_MIN,PRICE_MAX), "price band")

    psqft = c.df["price_usd"] / c.df["sqft"]
    c.keep(~(c.df["sqft"].notna() & (psqft > PSQFT_MAX)), "$/sqft sane")

    c.df["borough"] = c.df["neighborhoods"].apply(extract_borough)
    c.keep(c.df["borough"].notna(), "has borough")

    df = c.df
    c.print()

    # ------------------------
    # FEATURE ENGINEERING (merged best)
    # ------------------------

    # bedrooms
    df["bedrooms_missing"] = df["bedrooms"].isna().astype(int)
    df["bedrooms"] = df.groupby(["borough","bathrooms"])["bedrooms"].transform(lambda s: s.fillna(s.median()))
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())

    # bathrooms
    df["bathrooms"] = df["bathrooms"].fillna(1)

    # sqft
    df["sqft_missing"] = df["sqft"].isna().astype(int)
    borough_med = df.groupby("borough")["sqft"].transform("median")
    df["sqft_filled"] = df["sqft"].fillna(borough_med).fillna(df["sqft"].median())

    df["log_sqft"] = np.log1p(df["sqft_filled"])

    # distance
    df["dist_km"] = haversine_km(
        df["latitude"], df["longitude"],
        NYC_CENTER_LAT, NYC_CENTER_LON
    )
    df["log_dist_km"] = np.log1p(df["dist_km"])

    # flags
    df["is_studio"] = (df["bedrooms"] == 0).astype(int)

    for ccol in ["no_fee","by_owner","featured"]:
        df[ccol] = df.get(ccol,0)
        df[ccol] = df[ccol].fillna(0).astype(int)

    # target
    df["log_price"] = np.log(df["price_usd"])

    return df

# ------------------------
# MODEL
# ------------------------
def build_X(df):
    X = pd.DataFrame(index=df.index)

    X["bedrooms"] = df["bedrooms"]
    X["bathrooms"] = df["bathrooms"]
    X["log_sqft"] = df["log_sqft"]
    X["log_dist_km"] = df["log_dist_km"]
    X["sqft_missing"] = df["sqft_missing"]
    X["is_studio"] = df["is_studio"]
    X["no_fee"] = df["no_fee"]
    X["by_owner"] = df["by_owner"]

    for b in BOROUGHS:
        if b == BASELINE_BOROUGH: continue
        X[f"is_{b.lower().replace(' ','_')}"] = (df["borough"]==b).astype(int)

    X = sm.add_constant(X)
    return X

def fit_and_report(df):
    X = build_X(df)
    y = df["log_price"]

    model = sm.OLS(y, X).fit(cov_type="HC3")
    print(model.summary())

    df = df.copy()
    df["pred_price"] = np.exp(model.predict(X))
    df["pct_vs_model"] = df["price_usd"]/df["pred_price"] - 1

    return model, df

# ------------------------
# CLI
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("listings.jsonl"))
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    df = load_and_clean(args.inp)
    model, df = fit_and_report(df)

    print("\nTop underpriced:")
    print(df.nsmallest(args.top_k,"pct_vs_model")[["price_usd","pred_price","pct_vs_model"]])

    print("\nTop overpriced:")
    print(df.nlargest(args.top_k,"pct_vs_model")[["price_usd","pred_price","pct_vs_model"]])

if __name__ == "__main__":
    main()