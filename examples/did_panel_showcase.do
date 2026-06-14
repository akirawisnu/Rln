* ============================================================
* Rln — Dynamic Difference-in-Differences Showcase
*
* Demonstrates every estimator in the diff-diff 3.x suite on a
* realistic STAGGERED panel: 120 units × 10 years (2014–2023),
* three treatment cohorts (2018, 2020, 2022), plus a never-treated
* control cohort. This is the setup where modern causal-inference
* estimators differ meaningfully from vanilla TWFE.
*
* Data: examples/panel_staggered.csv  (shipped with Rln)
*   state_id        — panel unit identifier
*   year            — time variable
*   ever_treated    — 1 if ever treated, 0 otherwise
*   first_treat_year— year treatment started (0 for never-treated)
*   treated         — 1 if this row is post-treatment for this unit
*   gdp_growth      — outcome
*   true_effect     — the actual treatment effect per row (for validation)
*
* Required: ssc install diff-diff
* ============================================================

global path "examples"

use "$path/panel_staggered.csv", clear
describe
count
xtset state_id year
summarize gdp_growth

* Sanity check — what does the "truth" look like?
tabulate first_treat_year
summarize true_effect if ever_treated == 1

* ============================================================
* 1. VANILLA TWFE — starts biased under heterogeneous effects
* ============================================================
* This is the textbook regression. With staggered adoption and
* heterogeneous effects it can be badly biased — the Goodman-Bacon
* decomposition (section 9) will show why.

didregress (gdp_growth) (treated), group(state_id) time(year) method(twfe) robust
coefplot, title("TWFE — single ATT") name("$path/did_01_twfe.png")

* ============================================================
* 2. BASIC 2x2 DiD (diff-diff backend)
* ============================================================
* Useful as a benchmark when data are NOT staggered. Here we'd
* essentially be pooling the cohorts; included for completeness.

didregress (gdp_growth) (treated), group(state_id) time(year) method(did)
coefplot, title("Basic 2x2 DiD") name("$path/did_02_did.png")

* ============================================================
* 3. CALLAWAY-SANT'ANNA (cs) — the workhorse staggered estimator
* ============================================================
* Compares each treated cohort against never-treated (or not-yet-treated)
* units at the correct pre- and post-treatment periods, avoiding the
* "forbidden comparisons" that bias TWFE.

didregress (gdp_growth) (treated), group(state_id) time(year) ///
    method(cs) first_treat(first_treat_year) aggregate(simple)
coefplot, title("Callaway-Sant'Anna — simple ATT") name("$path/did_03_cs_simple.png")

* Event-study flavor — dynamic effects by relative time
didregress (gdp_growth) (treated), group(state_id) time(year) ///
    method(cs) first_treat(first_treat_year) aggregate(event_study)
coefplot, title("Callaway-Sant'Anna — event study") name("$path/did_04_cs_es.png")

* ============================================================
* 4. SUN-ABRAHAM (sa) — interaction-weighted regression alternative
* ============================================================
* Cross-check for CS. If CS and SA agree, results are credible.

didregress (gdp_growth) (treated), group(state_id) time(year) ///
    method(sa) first_treat(first_treat_year)
coefplot, title("Sun-Abraham") name("$path/did_05_sa.png")

* ============================================================
* 5. BORUSYAK-JARAVEL-SPIESS (bjs) — efficient imputation DiD
* ============================================================
* The MOST efficient estimator under homogeneous treatment effects.
* Produces ~50% shorter CIs than CS if parallel trends holds.
*
* The aliases method(did_imputation), method(borusyak), and
* method(imputation) all route to the same estimator.

didregress (gdp_growth) (treated), group(state_id) time(year) ///
    method(bjs) first_treat(first_treat_year) aggregate(simple)
coefplot, title("Borusyak-Jaravel-Spiess (did_imputation)") name("$path/did_06_bjs.png")

* ============================================================
* 6. GARDNER (2022) — two-stage DiD
* ============================================================
* Same point estimate as BJS but with a GMM sandwich variance
* that accounts for first-stage uncertainty — often a more
* defensible standard error.

didregress (gdp_growth) (treated), group(state_id) time(year) ///
    method(gardner) first_treat(first_treat_year)
coefplot, title("Gardner two-stage DiD") name("$path/did_07_gardner.png")

* ============================================================
* 7. STACKED DIFF-IN-DIFF (stacked) — Wing, Freedman, Hollingsworth 2024
* ============================================================
* Builds a "clean" sub-experiment for each treated cohort and stacks
* them — closer to how practitioners intuitively want event studies
* to work.

didregress (gdp_growth) (treated), group(state_id) time(year) ///
    method(stacked) first_treat(first_treat_year) aggregate(event_study)
coefplot, title("Stacked DiD — event study") name("$path/did_08_stacked.png")

* ============================================================
* 8. SYNTHETIC DIFF-IN-DIFF (sdid) — few treated units, long pre-period
* ============================================================
* Reweights control units to optimally match treated pre-trends.
* Especially useful when a single cohort dominates or there are
* concerns about the parallel-trends assumption.

didregress (gdp_growth) (treated), group(state_id) time(year) ///
    method(sdid) post_periods(2018 2019 2020 2021 2022 2023)
coefplot, title("Synthetic DiD") name("$path/did_09_sdid.png")

* ============================================================
* 9. GOODMAN-BACON DECOMPOSITION (bacon) — diagnose TWFE bias
* ============================================================
* Splits the TWFE estimate into its constituent 2x2 comparisons and
* reports how much weight each gets. "Later vs earlier" weights are
* the dangerous ones — they subtract *already-treated* cohorts from
* newly-treated ones.
*
* Note: bacon is a DIAGNOSTIC, not an estimator with a single ATT
* that lends itself to coefplot. Read its decomposition table directly.

didregress (gdp_growth) (treated), group(state_id) time(year) ///
    method(bacon) first_treat(first_treat_year)

* ============================================================
* 10. EVENT STUDY with pre- and post- period effects
* ============================================================
* For diagnosing parallel trends: effects in pre-treatment periods
* should be ≈ 0 if the assumption holds. The reference period's
* coefficient is zero by construction.
*
* This is the canonical case for coefplot — the resulting figure
* shows leads, lags, and the reference period in one frame.

didregress (gdp_growth) (treated), group(state_id) time(year) ///
    method(eventstudy) ///
    post_periods(2018 2019 2020 2021 2022 2023) ///
    reference_period(2017)
coefplot, title("Event study — leads & lags") name("$path/did_10_eventstudy.png")

* ============================================================
* 11. HONEST DIFF-IN-DIFF sensitivity (Rambachan & Roth 2023)
* ============================================================
* Even if pre-trends look parallel, they might have low power.
* Honest DiD asks: how big could a post-treatment parallel-trends
* violation be before our conclusion flips? M=1 says the violation
* cannot exceed the largest pre-treatment violation.
*
* Must be called AFTER a method(eventstudy) run — Rln stores the
* event-study results and reuses them here. Honest produces a
* sensitivity table; coefplot is not applicable.

didregress (gdp_growth) (treated), method(honest) m(0.5)
didregress (gdp_growth) (treated), method(honest) m(1.0)
didregress (gdp_growth) (treated), method(honest) m(2.0)

* If the conclusion holds even at M=2.0, it is very robust.

* ============================================================
* Summary workflow recommendation (from the diff-diff practitioner guide):
*
*   1. Run TWFE + BACON to see if bias is a concern
*   2. If yes, run CS and SA as main specifications
*   3. Run BJS/GARDNER for tighter CIs (efficient estimators)
*   4. Run EVENTSTUDY to visualize dynamics + check parallel trends
*   5. Use coefplot after each method to compare results visually
*   6. Run HONEST for sensitivity-to-violations diagnostics
*   7. If your panel has ≤ 5 treated units, use SDID instead of CS
* ============================================================
