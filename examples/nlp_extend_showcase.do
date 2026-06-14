* ============================================================
* Rln NLP-Extend + Causal-Inference Showcase
*   - Offline translate  (argos-translate, portable models)
*   - Offline summarize  (sumy, pure-Python extractive)
*   - diff-diff 3.x methods: CS / SA / BJS / Gardner / Stacked /
*     SynthDiD / event study / Goodman-Bacon / Honest DiD
*
* All `hf ...` syntax from the old showcase still works — `hf` is
* now an alias for `nlp`, so legacy do-files continue unchanged.
* ============================================================

global path "examples"

* ============================================================
* PART A. OFFLINE TRANSLATION (argos-translate)
* ============================================================
*
* SETUP (run once, requires internet):
*   nlp download translate id en
*   nlp download translate en de
*   nlp download translate en id
*
* After that, everything below runs OFFLINE on any machine you
* copy the rln/ folder to (Windows / macOS / Linux).

use "$path/nlp_survey.csv", clear
describe

* Direct pair
nlp translate text_indonesian, from(id) to(en) generate(text_id_en)
list text_indonesian text_id_en in 1/3

* Pivot: id -> en -> de  (no direct id->de model needed)
nlp translate text_indonesian, from(id) to(de) generate(text_id_de)
list text_indonesian text_id_de in 1/3

nlp translate article, from(en) to(de) generate(article_de)

* ============================================================
* PART B. OFFLINE EXTRACTIVE SUMMARIZATION (sumy)
* ============================================================
*
* No download needed — sumy ships with the requirements. A tiny
* offline tokenizer + stopword list means even in a locked-down
* network sandbox the summarizer still works.

* Default: LSA, 3 sentences
nlp summarize article, generate(article_sum_lsa)

* TextRank with 2 sentences
nlp summarize article, generate(article_sum_tr) method(textrank) sentences(2)

* LexRank with 5 sentences, German stopwords (works on German text)
nlp summarize article_de, generate(article_de_sum) method(lexrank) ///
    sentences(5) language(german)

gen art_len = strlen(article)
gen sum_len = strlen(article_sum_lsa)
summarize art_len sum_len

* Legacy-style call also works — maxlen() is silently translated
* into sentences() so existing 'hf summarize ..., maxlen(120)' do-files keep running:
nlp summarize article, generate(article_sum_legacy) maxlen(120) minlen(20)

* ============================================================
* PART C. EVERYTHING ELSE FROM THE OLD hf DISPATCHER STILL WORKS
* ============================================================
* These delegate to the HuggingFace backend (nlp.py) unchanged.

nlp classify survey_response, labels("economy healthcare education environment") ///
    generate(policy_topic)
nlp sentiment product_review, generate(review_sent)
nlp ner entity_text, generate(entities)
nlp cache

* ============================================================
* Done!  See README.md for the full command reference.
* ============================================================
