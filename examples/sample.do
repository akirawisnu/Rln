* ============================================================
* Rln Sample Do-File
* Demonstrates data loading, exploration, and transformation
* ============================================================

* Load the dataset
global path "examples"
clear
use "$path/mydata.csv"

* --- Exploration ---
describe
summarize
count

* --- Variable Management ---
generate age_sq = age^2
generate ln_income = ln(income) if income > 0
label variable age_sq "Age squared"
label variable ln_income "Log of monthly income"

* --- Recode example ---
recode education (0/8=1) (9/12=2) (13/20=3), generate(ed_level)
label define ed_lbl 1 "Primary" 2 "Secondary" 3 "Tertiary"
label values ed_level ed_lbl

* --- Data cleaning ---
duplicates report id
drop if age < 0
drop if missing(income)
keep if age >= 18 & age <= 65

* --- Sorting ---
sort id

* --- Tabulations ---
tabulate city
tabulate city employed
summarize income, detail

* --- Merge with demographics ---
merge 1:1 id using "$path/demographics.csv"
tabulate _merge

* Keep only matched
keep if _merge == 3
drop _merge

* --- Final summary ---
describe
count
tabulate ed_level
tabulate gender

* --- Regression ---
regress ln_income age age_sq education
ereturn list
predict yhat
predict resid, residuals

histogram resid
graph export "$path/resid_hist.png", replace

* --- Loops ---
local v "income education age"
display "`v'"

foreach v in income education age {
    summarize `v'
}


* --- Export ---
save "$path/cleaned_data.dta", replace
export delimited "$path/cleaned_data.csv", replace

* Done!
