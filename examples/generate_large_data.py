"""
Generate synthetic large dataset for LRTM (Larger-than-RAM) testing.
Creates a 2M row dataset in Parquet format (~200MB).

Usage: python generate_large_data.py [nrows]
Default: 2,000,000 rows
"""

import sys
import os
import time

def generate(n_rows=2_000_000, output_dir="."):
    try:
        import polars as pl
    except ImportError:
        print("Polars required: pip install polars")
        return

    import numpy as np
    np.random.seed(2026)

    print(f"Generating {n_rows:,} rows...")
    t0 = time.time()

    # Main dataset: firm-level panel data (firms x years)
    n_firms = n_rows // 10
    years = list(range(2015, 2025))

    firm_ids = np.repeat(np.arange(1, n_firms + 1), len(years))[:n_rows]
    year_col = np.tile(years, n_firms)[:n_rows]

    # Firm characteristics (stable within firm, vary across)
    firm_size = np.random.choice(["micro", "small", "medium", "large"], n_firms, p=[0.4, 0.3, 0.2, 0.1])
    firm_sector = np.random.choice(
        ["manufacturing", "services", "agriculture", "technology", "retail",
         "finance", "healthcare", "construction", "transport", "energy"],
        n_firms
    )
    firm_country = np.random.choice(
        ["DE", "FR", "IT", "ES", "NL", "PL", "SE", "AT", "BE", "CZ",
         "PT", "DK", "FI", "IE", "GR", "RO", "HU", "SK", "HR", "BG"],
        n_firms, p=[0.15, 0.13, 0.12, 0.10, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03,
                    0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.02]
    )
    firm_region = np.random.choice(
        ["Western", "Southern", "Northern", "Eastern", "Central"],
        n_firms, p=[0.35, 0.25, 0.12, 0.13, 0.15]
    )

    # Expand firm-level to panel
    size_col = np.repeat(firm_size, len(years))[:n_rows]
    sector_col = np.repeat(firm_sector, len(years))[:n_rows]
    country_col = np.repeat(firm_country, len(years))[:n_rows]
    region_col = np.repeat(firm_region, len(years))[:n_rows]

    # Time-varying outcomes
    revenue = np.random.lognormal(mean=12, sigma=2, size=n_rows).round(2)
    employees = np.random.lognormal(mean=3, sigma=1.5, size=n_rows).astype(int).clip(1, 50000)
    productivity = (revenue / employees).round(2)
    export_share = np.random.beta(2, 5, size=n_rows).round(4)
    rd_intensity = np.random.beta(1.5, 20, size=n_rows).round(4)
    ai_adopted = np.random.binomial(1, 0.15, size=n_rows)
    digital_score = np.random.uniform(0, 100, size=n_rows).round(1)
    profit_margin = np.random.normal(0.08, 0.15, size=n_rows).round(4)
    debt_ratio = np.random.beta(3, 5, size=n_rows).round(4)
    age_firm = np.random.randint(1, 100, size=n_rows)

    print(f"  Building DataFrame...")

    df = pl.DataFrame({
        "firm_id": firm_ids.astype(np.int32),
        "year": year_col.astype(np.int16),
        "country": country_col,
        "region": region_col,
        "sector": sector_col,
        "firm_size": size_col,
        "revenue": revenue.astype(np.float32),
        "employees": employees.astype(np.int32),
        "productivity": productivity.astype(np.float32),
        "export_share": export_share.astype(np.float32),
        "rd_intensity": rd_intensity.astype(np.float32),
        "ai_adopted": ai_adopted.astype(np.int8),
        "digital_score": digital_score.astype(np.float32),
        "profit_margin": profit_margin.astype(np.float32),
        "debt_ratio": debt_ratio.astype(np.float32),
        "firm_age": age_firm.astype(np.int16),
    })

    # Save as parquet
    parquet_path = os.path.join(output_dir, "eu_firms_panel.parquet")
    df.write_parquet(parquet_path)
    parquet_size = os.path.getsize(parquet_path) / 1e6

    # Create a smaller reference dataset for merging
    ref = pl.DataFrame({
        "country": ["DE", "FR", "IT", "ES", "NL", "PL", "SE", "AT", "BE", "CZ",
                     "PT", "DK", "FI", "IE", "GR", "RO", "HU", "SK", "HR", "BG"],
        "country_name": ["Germany", "France", "Italy", "Spain", "Netherlands", "Poland",
                         "Sweden", "Austria", "Belgium", "Czech Republic", "Portugal",
                         "Denmark", "Finland", "Ireland", "Greece", "Romania",
                         "Hungary", "Slovakia", "Croatia", "Bulgaria"],
        "eu_region": ["Western", "Western", "Southern", "Southern", "Western", "Eastern",
                      "Northern", "Western", "Western", "Eastern", "Southern",
                      "Northern", "Northern", "Western", "Southern", "Eastern",
                      "Eastern", "Eastern", "Southern", "Eastern"],
        "gdp_per_capita": [51203, 43518, 36812, 30115, 57767, 17318, 55566, 53267, 49558, 26379,
                           24563, 67218, 53654, 100172, 20867, 14858, 18075, 19266, 17399, 12221],
    })
    ref_path = os.path.join(output_dir, "eu_country_ref.parquet")
    ref.write_parquet(ref_path)

    elapsed = time.time() - t0
    print(f"\nGenerated:")
    print(f"  {parquet_path}: {n_rows:,} rows x {len(df.columns)} cols ({parquet_size:.1f} MB)")
    print(f"  {ref_path}: {len(ref)} rows (reference data)")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    output = sys.argv[2] if len(sys.argv) > 2 else "."
    generate(n, output)
