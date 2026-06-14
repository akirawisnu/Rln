* ============================================================
* Rln — other statistical tools Functions Showcase
*   inlist, inrange, regexm, regexr, regexs, real, string, length
*
* Runs against examples/demographics.csv.
* All functions work identically inside gen / replace / if / keep / drop.
* ============================================================

global path "examples"

use "$path/demographics.csv", clear
describe
list in 1/5

* ============================================================
* 1. inlist() — set-membership check (replaces verbose `if x==v1 | x==v2`)
* ============================================================

* Old, verbose way:
*   gen big = (city == "New York" | city == "Los Angeles" | city == "Chicago")
* New:
gen bigcity = inlist(city, "New York", "Los Angeles", "Chicago")
tabulate bigcity

* inlist also works on numbers:
gen young_adult = inlist(age, 18, 19, 20, 21, 22, 23, 24, 25)
tabulate young_adult

* Use in `if` directly:
summarize income if inlist(gender, "Male", "Female")

* ============================================================
* 2. inrange() — interval membership (inclusive)
* ============================================================

gen workingage = inrange(age, 18, 65)
tabulate workingage

* first arg can be an EXPRESSION, not just a bare variable:
gen decade_born = inrange(2025 - age, 1970, 1979)
tabulate decade_born

* Use with filter:
count if inrange(income, 30000, 80000)
summarize income if inrange(age, 25, 45)

* ============================================================
* 3. regexm() — regex match (returns 1/0)
* ============================================================

* Detect a pattern anywhere in a string:
gen has_digit_in_name = regexm(name, "[0-9]")
tabulate has_digit_in_name

* Validate something like an email:
gen valid_email = regexm(email, "^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$")
tabulate valid_email

* Filter with regex:
list name email if !regexm(email, "@") in 1/5

* ============================================================
* 4. regexr() — regex replace
* ============================================================

* Strip all digits from a field:
gen name_clean = regexr(name, "[0-9]+", "")

* Mask phone numbers in a free-text column:
gen notes_masked = regexr(notes, "[0-9]{3}-[0-9]{4}", "XXX-XXXX")

* Normalize whitespace:
gen name_norm = regexr(name, "\s+", " ")

list name name_clean in 1/5

* ============================================================
* 5. regexs() — extract a capture group (1-based, as documented here)
* ============================================================

* Get the area code out of "(212) 555-1234"-style phone numbers:
gen area_code = regexs(phone, "\(([0-9]+)\)", 1)

* Get the username portion of an email:
gen email_user = regexs(email, "^([^@]+)@", 1)

* Get the domain portion:
gen email_domain = regexs(email, "@(.+)$", 1)

list email email_user email_domain in 1/5

* Non-matching rows return missing — clean up with `if`:
count if missing(email_domain)

* ============================================================
* 6. Type conversion: real() and string()
* ============================================================

* `real()` converts a string column to numeric. Unparseable values become missing.
* Useful when a CSV loaded a numeric column as text:
gen income_str = string(income)
gen income_back = real(income_str)
summarize income income_back

* `string()` goes the other way:
gen year_str = string(year_of_birth)

* ============================================================
* 7. length() — character count (alias for strlen)
* ============================================================

gen name_len = length(name)
summarize name_len, detail

* Useful for filtering out abnormally short/long free-text:
keep if inrange(length(name), 2, 60)

* ============================================================
* 8. Combining the new functions — compound conditions
* ============================================================

* Three-predicate compound:
count if inrange(age, 25, 45)          ///
       & inlist(city, "New York", "Boston", "San Francisco") ///
       & regexm(email, "\.edu$")

* Regex OR set-membership:
gen tech_worker = inlist(job_title, "Engineer", "Developer") ///
                | regexm(job_title, "(?i)software|data scientist")
tabulate tech_worker

* ============================================================
* Done! All functions work inside gen, replace, if, keep, drop, count,
* summarize, and anywhere else Rln evaluates an expression.
* ============================================================
