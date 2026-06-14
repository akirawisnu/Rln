* ============================================================
* Rln NLP Showcase Do-File — NEURAL (HuggingFace) backend
* Demonstrates HuggingFace integration with nlp_survey.csv
*
* This showcase uses the NEURAL backend (`hf` / `nlp classify|sentiment|ner|embed`)
* which downloads HuggingFace transformer models to rln/hf_models/.
*
* For the OFFLINE-FIRST alternative that uses argos-translate and sumy
* (no neural weights, pure Python, works on any machine with no internet
* after the first download), see: examples/nlp_extend_showcase.do
*
* Side-by-side summary:
*
*  TASK         | Neural (this file)      | Offline (nlp_extend_showcase.do)
*  -------------+-------------------------+-----------------------------------
*  Translate    | Helsinki-NLP / NLLB      | argos-translate (CTranslate2 + OpenNMT)
*  Summarize    | BART / mT5 (abstractive) | sumy LSA/LexRank/TextRank (extractive)
*  Classify     | BART-MNLI zero-shot      | (use this file — no offline equivalent)
*  Sentiment    | multilingual BERT        | (use this file — no offline equivalent)
*  NER          | BERT / xlm-roberta       | (use this file — no offline equivalent)
*  Embed        | sentence-transformers    | (use this file — no offline equivalent)
*
* ============================================================
*
* SETUP (run once, needs internet):
*   hf download facebook/bart-large-mnli
*   hf download nlptown/bert-base-multilingual-uncased-sentiment
*   hf download facebook/bart-large-cnn
*   hf download facebook/nllb-200-distilled-600M
*   hf download dslim/bert-base-NER
* ============================================================

global path "examples"

* --- Load data ---
use "$path/nlp_survey.csv", clear
describe
count

* ============================================================
* 1. ZERO-SHOT TEXT CLASSIFICATION
*    Classify open-ended survey responses into policy topics
* ============================================================

hf classify survey_response, labels("economy healthcare education environment security") generate(policy_topic)
tabulate policy_topic
tabulate policy_topic respondent_gender
tabulate policy_topic country

* ============================================================
* 2. SENTIMENT ANALYSIS (multilingual, 1-5 stars)
* ============================================================

hf sentiment product_review, generate(review_sentiment)
tabulate review_sentiment
summarize review_sentiment_score, detail

* ============================================================
* 3. TRANSLATION — Indonesian and German to English and Russian
*    Using NLLB-200 (200 languages, works offline after download)
* ============================================================

hf translate text_indonesian, from(id) to(en) generate(text_id_english) model(facebook/nllb-200-distilled-600M)
hf translate text_german, from(de) to(en) generate(text_de_english) model(facebook/nllb-200-distilled-600M)
hf translate text_indonesian, from(id) to(ru) generate(text_id_russian) model(facebook/nllb-200-distilled-600M)
hf translate text_german, from(de) to(ru) generate(text_de_russian) model(facebook/nllb-200-distilled-600M)

* View sample translations
list text_indonesian text_id_english in 1/3

* ============================================================
* 4. TEXT SUMMARIZATION
* ============================================================

hf summarize article, generate(article_summary) maxlen(80) minlen(20)

* Compare lengths
gen article_len = strlen(article)
gen summary_len = strlen(article_summary)
summarize article_len summary_len

* ============================================================
* 5. NAMED ENTITY RECOGNITION
* ============================================================

hf ner entity_text, generate(entities)
summarize entities_count, detail
list entity_text entities in 1/5

* ============================================================
* 6. CROSS-TABULATION OF NLP RESULTS WITH DEMOGRAPHICS
* ============================================================

describe
count
tabulate country review_sentiment
tabulate year policy_topic

* ============================================================
* 7. EXPORT
*    CSV works best for multilingual text (Russian, etc.)
*    DTA uses other statistical tools 14+ format (version 118) for Unicode
* ============================================================

export delimited "$path/nlp_survey_enriched.csv", replace
save "$path/nlp_survey_enriched.dta", replace

* Done!
