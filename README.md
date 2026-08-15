# What Patients Really Can't Forgive: Mining 2,000 Drug Reviews with an LLM

**Finding:** Side effects that touch mood and self-image (depression, acne, weight gain) drag patient satisfaction down 1.3 to 3 points, far more than physical side effects like nausea. Patients tolerate feeling briefly sick; they don't forgive a drug that changes their mood or their body.

## The question

Patient drug reviews are rich but unstructured: thousands of free-text paragraphs that no one can analyze at scale. Buried in them is a question worth answering: which side effects actually make patients rate a drug poorly? Not which are most common, but which ones matter.

## What I built

A Python pipeline that:

- **Extracts structure from free text.** It uses an LLM (Claude) to read each review and pull side effects, sentiment, and whether the drug helped, returning clean JSON.
- **Builds an analyzable dataset.** 2,000 messy reviews become a structured table.
- **Finds the insight.** It filters and aggregates to measure how each side effect moves the average rating.
- **Predicts dissatisfaction.** A logistic-regression model flags low-rated reviews from the extracted features.

## Key results

- **The insight:** across 2,000 reviews, mood and appearance side effects (depression, average rating 3.9; acne, 4.4; weight gain, 5.7) scored far below the 6.97 overall average, while physical effects like nausea (6.4) barely moved it.
- **Validated, not assumed:** LLM-labeled sentiment separated star ratings by about 6.5 points (positive reviews averaged 9.1, negative 2.6), confirming the extraction tracks reality.
- **Held up under scale:** the pattern survived scaling from 500 to 2,000 reviews, ruling out small-sample noise.
- **The model:** predicts patient dissatisfaction at 0.79 recall and 0.86 precision, above the roughly 70% naive baseline, with recall chosen as the key metric given the class imbalance (about 30% low-rated).

## What made it interesting

- **Human language, not keywords:** the LLM catches "I couldn't sleep" as insomnia, something a keyword search would miss entirely.
- **Honest evaluation:** I used precision and recall instead of accuracy, because accuracy is misleading on imbalanced data.

## Tech

Python, pandas, scikit-learn, Anthropic API (Claude), Google Colab.

## How to run

1. Install dependencies: pip install -r requirements.txt
2. Run the notebook top to bottom, or load the included extracted_reviews_2000.csv to skip the extraction step.

## Data

Public UCI Drug Review Dataset (patient reviews from Drugs.com).
