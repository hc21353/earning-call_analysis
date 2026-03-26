# Earnings Call Transcript Analysis with FinBERT

## Overview

This project analyzes earnings call transcripts using FinBERT to extract sentiment signals and uncover insights from corporate communication.

The analysis goes beyond simple sentiment scoring by introducing interpretable features such as tone gap, uncertainty, and sentiment volatility.


## Dataset

* Source: Motley Fool (scraped earnings call transcripts) from Kaggle
* Size: ~18,000 transcripts
* Period: 2019–2022
* Content: Prepared remarks + Q&A sections


## Methodology

### 1. Text Processing

* Split transcripts into:

  * Prepared Remarks (management statements)
  * Q&A Section (analyst interaction)
* Sentence-level tokenization to avoid transformer input limits

### 2. Sentiment Analysis

* Model: FinBERT (`yiyanghkust/finbert-tone`)
* Approach:

  * Sentence-level inference
  * Mean aggregation across sentences

### 3. Feature Engineering

* **Net Sentiment Score**: positive − negative
* **Tone Gap**: difference between prepared remarks and Q&A
* **Uncertainty Score**: neutral probability
* **Sentiment Volatility**: standard deviation of sentence sentiment
* **High Negativity Flag**: Q&A negativity threshold

### 4. Text Analysis

* Lexicon-based:

  * Uncertainty words
  * Hedging language
* TF-IDF:

  * Distinguishing terms between positive vs negative calls

### 5. Sector-Level Analysis

* Sector mapping via yfinance
* Weighted sentiment (market cap-based)
* Trend analysis using linear regression
* Sector classification:

  * Growing / Stable / Cyclical / Declining
