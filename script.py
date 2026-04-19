import numpy as np
import pandas as pd

INPUT_FILE = "cars.csv"
OUTPUT_FILE = "cleaned_car_dataset.csv"


def budget_category(price_lakhs: float) -> str:
    if price_lakhs < 8:
        return "budget"
    if price_lakhs < 20:
        return "mid"
    return "premium"


def preprocess() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.lower().str.strip()

    column_map = {
        "brand": "make",
        "car_name": "model",
        "price_lakhs": "price",
        "rating": "user_rating",
        "safety_stars": "safety_rating",
        "mileage_kmpl": "mileage",
        "power_bhp": "power",
        "sales_fy2024": "sales",
    }
    df = df.rename(columns=column_map)

    required = [
        "make",
        "model",
        "price",
        "user_rating",
        "safety_rating",
        "mileage",
        "power",
        "sales",
    ]

    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df = df[required].copy()

    numeric_cols = ["price", "user_rating", "safety_rating", "mileage", "power", "sales"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fill_defaults = {
        "user_rating": 4.2,
        "safety_rating": 3.5,
        "mileage": 16.0,
        "power": 110.0,
        "sales": 10000.0,
    }

    for col in numeric_cols:
        if col == "price":
            continue
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = fill_defaults[col]
        df[col] = df[col].fillna(median_val)

    df = df.dropna(subset=["price", "make", "model"])

    # Keep broad, realistic range for Indian market data.
    df = df[(df["price"] >= 3) & (df["price"] <= 400)]
    df = df[(df["user_rating"] >= 1) & (df["user_rating"] <= 5)]
    df = df[(df["safety_rating"] >= 0) & (df["safety_rating"] <= 5)]
    df = df[(df["mileage"] >= 5) & (df["mileage"] <= 70)]

    df = df.drop_duplicates(subset=["make", "model"])

    # Balanced score for first-pass ranking.
    df["value_score"] = (
        0.30 * df["user_rating"]
        + 0.25 * df["safety_rating"]
        + 0.15 * (df["mileage"] / 10.0)
        + 0.15 * (df["power"] / 100.0)
        + 0.15 * np.log10(df["sales"] + 1)
    )

    df = df.sort_values("value_score", ascending=False).reset_index(drop=True)

    # Keep display-friendly numeric precision in lakhs/kmpl/stars.
    df["price"] = (df["price"] * 100000).round().astype(int)
    df["mileage"] = df["mileage"].round(1)
    df["safety_rating"] = df["safety_rating"].round(1)
    df["power"] = df["power"].round(1)

    # Build requested fields that are not directly available in the source.
    df["variant"] = "standard"
    df["user_rating"] = df["user_rating"].round(1)

    # Keep only requested output schema.
    output_cols = [
        "make",
        "model",
        "variant",
        "price",
        "mileage",
        "power",
        "safety_rating",
        "user_rating",
    ]
    return df[output_cols].copy()


def main() -> None:
    cleaned = preprocess()
    try:
        cleaned.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {len(cleaned)} rows to {OUTPUT_FILE}")
    except PermissionError:
        fallback_file = "cleaned_car_dataset_updated.csv"
        cleaned.to_csv(fallback_file, index=False)
        print(
            f"{OUTPUT_FILE} is in use. Saved {len(cleaned)} rows to {fallback_file} instead."
        )


if __name__ == "__main__":
    main()
