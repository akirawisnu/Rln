* ============================================================
* Rln LRTM Showcase — Larger-than-RAM Data Processing
* Uses Polars for memory-efficient analysis of large datasets
* ============================================================
*
* SETUP: Generate test data first (requires polars):
*   cd examples
*   python generate_large_data.py 2000000
*
* This creates:
*   eu_firms_panel.parquet  — 2M rows panel data (~80MB)
*   eu_country_ref.parquet  — 20 rows country reference
*
* Alternatively, convert existing data:
*   lrtm convert "big_data.csv"
*   lrtm convert "survey.dta", output("survey.parquet")
* ============================================================

global path "examples"

* ============================================================
* 1. CONVERT — One-time conversion from CSV/DTA/Excel to Parquet
*    After this, lazy loading is instant
* ============================================================

* Convert the CSV sample data to Parquet for demonstration
* (skip if you already generated with generate_large_data.py)
* lrtm convert "$path/mydata.csv", output("$path/mydata.parquet")

* ============================================================
* 2. LAZY LOAD — Zero RAM, just reads Parquet metadata
* ============================================================

lrtm use "$path/eu_firms_panel.parquet"
lrtm status
lrtm describe

* ============================================================
* 3. SUMMARY STATISTICS — Streaming, reads only requested columns
*    RAM usage: ~24MB for 3 Float32 columns x 2M rows
* ============================================================

lrtm summarize revenue employees productivity
lrtm summarize export_share rd_intensity digital_score

* With if condition: only reads matching rows
lrtm summarize revenue employees if country == "DE"
lrtm summarize productivity if ai_adopted == 1

* Compound conditions with & and |
lrtm summarize revenue if country == "DE" & year >= 2022
lrtm summarize profit_margin if firm_size == "large" | firm_size == "medium"

* ============================================================
* 4. TABULATIONS — One-way and two-way (crosstab)
*    Reads only the tabulated column(s)
* ============================================================

lrtm tabulate country
lrtm tabulate sector
lrtm tabulate firm_size

* With if condition
lrtm tabulate sector if country == "DE"
lrtm tabulate firm_size if ai_adopted == 1

* Two-way crosstab
lrtm tabulate country firm_size
lrtm tabulate sector ai_adopted if country == "FR"

* ============================================================
* 5. FAST COUNTS WITH CONDITIONS
*    For Parquet without filter: reads metadata only (0 RAM)
* ============================================================

lrtm count
lrtm count if ai_adopted == 1
lrtm count if revenue > 1000000
lrtm count if country == "DE" & year >= 2022
lrtm count if firm_size == "large" & sector == "technology"

* ============================================================
* 6. LAZY TRANSFORMATIONS — Zero RAM, adds to query plan
* ============================================================

lrtm generate log_revenue = ln(revenue)
lrtm generate rev_per_emp = revenue / employees

* Conditional generate: only sets value where condition is true
lrtm generate high_revenue = revenue if revenue > 500000
lrtm generate de_productivity = productivity if country == "DE"

* ============================================================
* 7. FILTERING — Zero RAM, adds filter to query plan
* ============================================================

lrtm keep if year >= 2020
lrtm count

* ============================================================
* 8. MERGE WITH REFERENCE DATA
* ============================================================

lrtm merge using "$path/eu_country_ref.parquet", on(country)
lrtm describe

* ============================================================
* 9. CONDITIONAL HEAD — Preview filtered subsets
* ============================================================

lrtm head 5 if country == "DE"
lrtm head 5 if sector == "technology" & ai_adopted == 1

* ============================================================
* 10. CONDITIONAL SAVE — Export subsets to Parquet
* ============================================================

lrtm save "$path/german_firms.parquet" if country == "DE"
lrtm save "$path/large_tech_firms.parquet" if firm_size == "large" & sector == "technology"

* ============================================================
* 11. GROUPED AGGREGATION (collapse) with if
* ============================================================

* Save current state first
lrtm save "$path/eu_firms_filtered.parquet"

* Reload and collapse
lrtm use "$path/eu_firms_filtered.parquet"
lrtm collapse (mean) revenue, by(country year)
lrtm head 20

* Collapse with condition
lrtm use "$path/eu_firms_filtered.parquet"
lrtm collapse (mean) revenue if ai_adopted == 1, by(sector)
lrtm head 10

* ============================================================
* 12. COLLECT TO PANDAS — For regression and charts
* ============================================================

lrtm use "$path/eu_firms_panel.parquet", sample(5000)
describe
summarize revenue employees productivity

* Regression on sample
regress productivity rd_intensity digital_score ai_adopted
predict yhat
predict resid, residuals
margins, dydx(*)

* Charts
histogram productivity if productivity < 500000, bins(40) normal title("Productivity Distribution") name("$path/productivity_hist.png")
scatter productivity rd_intensity if productivity < 1000000, lfit title("Productivity vs R&D") name("$path/prod_vs_rd.png")

* Cross-tabs
tabulate sector ai_adopted
tabulate country firm_size

* ============================================================
* 13. CLEAR LRTM — Free RAM without exiting Rln
* ============================================================

lrtm clear

* ============================================================
* 14. EXPORT SAMPLE
* ============================================================

save "$path/lrtm_sample.dta", replace
export delimited "$path/lrtm_sample.csv", replace

* Done!
