"""
Earnings Call Transcript Analysis with FinBERT
Data: Earnings call transcripts scraped from Motley Fool
"""

# 데이터 로드 및 기본 EDA
import kagglehub
import pandas as pd
import numpy as np
import os
import re
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, linregress
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
import yfinance as yf

# 데이터셋 다운로드
# Motley Fool에서 스크래핑한 미국 상장기업 earnings call transcript
# 약 18,755개 transcript, 2019~2022년
path = kagglehub.dataset_download("tpotterer/motley-fool-scraped-earnings-call-transcripts")
file_path = os.path.join(path, "motley-fool-data.pkl")
df = pd.read_pickle(file_path)

print(f"전체 레코드 수: {len(df):,}")
print(f"컬럼: {df.columns.tolist()}")
df.head(3)


# 한글 폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False

# 날짜 파싱: pd.to_datetime(format='mixed')으로 변환 후 year, quarter 파생변수 생성
# year_q: '2020-Q2' 형태 그대로 유지 → 시계열 인덱스로 활용
# q 컬럼: '2020-Q2' → quarter 추출
df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
df['year'] = df['date_parsed'].dt.year
df['quarter'] = df['q'].str.extract(r'(Q\d)')
df['year_q']  = df['q']  # '2020-Q2' 형태 그대로 활용


# Transcript 관련 통계
# word_count: transcript를 공백 기준으로 split한 단어 수
# transcript 길이의 극단값 탐지
df['word_count'] = df['transcript'].apply(lambda x: len(str(x).split()))

print("=== Transcript Word Count ===")
print(df['word_count'].describe().round(1))

print("\n=== 연도별 transcript 수 ===")
print(df['year'].value_counts().sort_index())

# EDA 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# 그래프 1) Word count 히스토그램
axes[0].hist(df['word_count'], bins=60, color='steelblue', edgecolor='white', linewidth=0.5)
axes[0].axvline(df['word_count'].median(), color='red', linestyle='--',
                label=f"Median: {df['word_count'].median():.0f}")
axes[0].set_title('Transcript Word Count Distribution')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')
axes[0].legend()
# 그래프 2) 연도별 transcript 수 막대 차트
year_cnt = df['year'].value_counts().sort_index()
axes[1].bar(year_cnt.index.astype(str), year_cnt.values, color='steelblue')
axes[1].set_title('Transcripts per Year')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.show()


# Transcript 구조 분석 
# transcript 구조: Prepared Remarks(CEO/CFO의 발표문)+ Q&A (투자자 질문 + 경영진 답변)
# 구분자가 없으면 prepared만 반환 (qa_section = None)
def split_transcript(text):
    if pd.isna(text):
        return None, None 

    match = re.search(
        r'(?:Questions?[\s-]+and[\s-]+Answers?(?:[\s-]+sessions?)?|Questions?[\s-]+comes?[\s-]+from)',
        text, re.IGNORECASE
    )
    if match:
        idx = match.start()
        return text[:idx].strip(), text[idx:].strip()
    return text.strip(), None

df['prepared_remarks'], df['qa_section'] = zip(*df['transcript'].apply(split_transcript))

# Prepared Remarks vs Q&A 분리 성공률 및 단어 수 분석
qa_coverage = df['qa_section'].notna().mean()
print(f"Q&A 섹션 분리 성공률: {qa_coverage:.1%}")

df['prepared_wc'] = df['prepared_remarks'].apply(lambda x: len(str(x).split()) if x else np.nan)
df['qa_wc']       = df['qa_section'].apply(lambda x: len(str(x).split()) if x else np.nan)

print("\n=== 섹션별 Word Count ===")
print(pd.DataFrame({
    'prepared_wc': df['prepared_wc'].describe(),
    'qa_wc'      : df['qa_wc'].describe()
}).round(1))


# FinBERT 고도화 
# Sentence-level Sentiment + 3-class 확률 보존
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

# FinBERT 모델(yiyanghkust/finbert-tone) 로드
MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # inference 모드 — dropout 비활성화

# 0=positive, 1=negative, 2=neutral
# → all_probs[:, 0] = positive 확률, [:, 1] = negative 확률, [:, 2] = neutral 확률
LABELS = ['positive', 'negative', 'neutral']
print(f"Using device: {device}")
print(f"Label order (0=pos, 1=neg, 2=neu): {LABELS}")

# Sentence-level batch inference 함수
# mean pooling over sentences 활용:
#   BERT 계열 모델 최대 입력 = 512 토큰 (≈370 단어)
#   earnings call 평균 ≈ 8,000 단어 → 단순 truncation 시 정보 손실
#   → 문장 단위 분리 후 각각 inference, 전체 평균으로 문서 감성 표현

# 반환 변수 설명
#   pos_mean   : 문장 평균 Positive 확률
#   neg_mean   : 문장 평균 Negative 확률
#   neu_mean   : 문장 평균 Neutral  확률 — neutral 높을수록 모호한 발표
#   net_score  : pos_mean - neg_mean — 핵심 감성 지표
#   pos_ratio  : dominant class가 Positive인 문장 비율
#   neg_ratio  : dominant class가 Negative인 문장 비율
#   sent_count : 분석된 문장 수
#   score_std  : 문장별 net_score의 표준편차 = sentiment volatility
def sentences_to_finbert(
    text: str,
    batch_size: int = 16,
    min_sent_len: int = 15
) -> dict | None:
    if not text or pd.isna(text):
        return None

    # 문장 분리 + 짧은 문장 필터링 (speaker 이름, 'Thank you.' 등 제거)
    sents = [s.strip() for s in sent_tokenize(text) if len(s.strip()) >= min_sent_len]
    if not sents:
        return None

    all_probs = []   # shape: (N_sentences, 3)

    for i in range(0, len(sents), batch_size):
        batch = sents[i : i + batch_size]

        # padding=True  : 배치 내 최장 문장 기준 동적 패딩
        # truncation=True: BERT 최대 토큰(512) 초과 방지
        enc = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits

        # softmax로 확률 변환 (dim=1: class 차원)
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # shape: (batch, 3)
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)   # shape: (N_sentences, 3)

    # 문장별 dominant class: argmax로 각 문장이 pos/neg/neu 중 무엇인지 결정
    dominant = np.argmax(all_probs, axis=1)   # 0=pos, 1=neg, 2=neu

    # 문장별 net score (pos - neg): 감성 변동성 계산에 사용
    sent_scores = all_probs[:, 0] - all_probs[:, 1]

    return {
        'pos_mean'  : float(all_probs[:, 0].mean()),
        'neg_mean'  : float(all_probs[:, 1].mean()),
        'neu_mean'  : float(all_probs[:, 2].mean()),
        'net_score' : float(sent_scores.mean()),
        'pos_ratio' : float((dominant == 0).mean()),
        'neg_ratio' : float((dominant == 1).mean()),
        'sent_count': len(sents),
        'score_std' : float(sent_scores.std()),
    }

# 전체 18,755건 중 300건 샘플로 분석
N_SAMPLE = 300
df_sample = df.sample(N_SAMPLE, random_state=42).copy().reset_index(drop=True) # random_state=42: 재현 가능성 확보

tqdm.pandas(desc="Prepared Remarks")
prep_results = df_sample['prepared_remarks'].progress_apply(sentences_to_finbert)

tqdm.pandas(desc="Q&A Section")
qa_results   = df_sample['qa_section'].progress_apply(sentences_to_finbert)

def expand_result(series, prefix): #dict 시리즈를 DataFrame으로 전개 후 prefix 부착
    """dict 시리즈를 DataFrame으로 전개 후 prefix 부착."""
    expanded = pd.json_normalize(series)
    expanded.columns = [f"{prefix}_{c}" for c in expanded.columns]
    return expanded

prep_df = expand_result(prep_results, 'prep')
qa_df   = expand_result(qa_results,   'qa')

df_sample = pd.concat([df_sample, prep_df, qa_df], axis=1)

print("생성된 sentiment 컬럼:")
print([c for c in df_sample.columns if c.startswith(('prep_', 'qa_'))])
df_sample[['ticker', 'prep_net_score', 'qa_net_score', 'prep_score_std', 'qa_score_std']].head()


# Sentiment 파생 피처 생성
# tone_gap = prep_net_score - qa_net_score
# uncertainty_score = prep_neu_mean
#   neutral 문장이 많다 = 모호한 표현·hedge 언어가 다수
# sentiment_volatility = prep_score_std
#   발표 내 감성 변동폭 ↑ → 긍정/부정 발언이 섞인 불안정한 어조
# high_negativity = Q&A에서 부정 문장이 15% 초과하는 콜 식별 플래그
df_sample['tone_gap'] = (
    df_sample['prep_net_score'] - df_sample['qa_net_score']
)

df_sample['uncertainty_score']    = df_sample['prep_neu_mean']
df_sample['sentiment_volatility'] = df_sample['prep_score_std']

df_sample['high_negativity'] = (
    df_sample['qa_neg_ratio'].fillna(0) > 0.15
).astype(int)

print("=== 파생 피처 기술통계 ===")
feat_cols = ['tone_gap', 'uncertainty_score', 'sentiment_volatility', 'high_negativity']
print(df_sample[feat_cols].describe().round(4))

# Prepared vs Q&A sentiment 산점도 + Tone Gap 분포
# 그래프 1) Prepared vs Q&A 산점도
#
# 그래프 2) Tone Gap 히스토그램
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

valid = df_sample.dropna(subset=['prep_net_score', 'qa_net_score'])
sc = axes[0].scatter(
    valid['prep_net_score'], valid['qa_net_score'],
    c=valid['tone_gap'], cmap='RdYlGn_r', alpha=0.7, s=30
)
lim = [-0.3, 0.8]
axes[0].plot(lim, lim, 'k--', linewidth=1, alpha=0.5, label='y=x (parity)')
axes[0].set_xlabel('Prepared Remarks Net Score')
axes[0].set_ylabel('Q&A Net Score')
axes[0].set_title('Prepared vs Q&A Sentiment\n(color = tone_gap, green=large gap)')
axes[0].legend()
plt.colorbar(sc, ax=axes[0], label='tone_gap')

axes[1].hist(valid['tone_gap'].dropna(), bins=30, color='steelblue', edgecolor='white')
axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
axes[1].axvline(valid['tone_gap'].mean(), color='red', linestyle='-',
                label=f"Mean: {valid['tone_gap'].mean():.3f}")
axes[1].set_title('Tone Gap Distribution\n(Prepared - Q&A net score)')
axes[1].set_xlabel('Tone Gap')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.show()

# 3-class 확률 구성 시각화
# 그래프: 스택 막대 차트 (상단) + 수치 표 (하단)
#   prep_net_score 기준 Top10 vs Bottom10 비교
#   수치 표: 각 기업의 pos/neg/neu_mean 값
top10    = df_sample.nlargest(10, 'prep_net_score')[['ticker', 'prep_pos_mean', 'prep_neg_mean', 'prep_neu_mean']]
bottom10 = df_sample.nsmallest(10, 'prep_net_score')[['ticker', 'prep_pos_mean', 'prep_neg_mean', 'prep_neu_mean']]
compare  = pd.concat([top10, bottom10]).set_index('ticker')

fig, (ax, ax_table) = plt.subplots(2, 1, figsize=(10, 9),
                                    gridspec_kw={'height_ratios': [3, 2]})

bar_w = 0.6
x = np.arange(len(compare))
ax.bar(x, compare['prep_pos_mean'], bar_w, label='Positive', color='#2ecc71')
ax.bar(x, compare['prep_neg_mean'], bar_w, bottom=compare['prep_pos_mean'], label='Negative', color='#e74c3c')
ax.bar(x, compare['prep_neu_mean'], bar_w,
       bottom=compare['prep_pos_mean'] + compare['prep_neg_mean'], label='Neutral', color='#95a5a6')
ax.set_xticks(x)
ax.set_xticklabels(compare.index, rotation=45, ha='right', fontsize=8)
ax.axvline(9.5, color='black', linestyle='--', linewidth=1)
ax.text(4.5, 1.03, 'Top 10 (most positive)', ha='center', fontsize=9)
ax.text(14.5, 1.03, 'Bottom 10 (most negative)', ha='center', fontsize=9)
ax.set_ylabel('Probability')
ax.set_title('FinBERT 3-class Probability Composition\n(Prepared Remarks)')
ax.legend(loc='lower right')

table_data = compare.copy()
table_data.columns = ['Positive', 'Negative', 'Neutral']
table_data['Group'] = ['Top10'] * 10 + ['Bottom10'] * 10
table_data = table_data[['Group', 'Positive', 'Negative', 'Neutral']]

cell_text  = [[row['Group'],
               f"{row['Positive']:.4f}",
               f"{row['Negative']:.4f}",
               f"{row['Neutral']:.4f}"]
              for _, row in table_data.iterrows()]
col_labels = ['Group', 'Positive', 'Negative', 'Neutral']

ax_table.axis('off')
tbl = ax_table.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc='center',
    loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.4)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, 21):
    color = '#eafaf1' if i <= 10 else '#fdecea'
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor(color)

plt.tight_layout()
plt.show()

# 텍스트 기반 Uncertainty/Hedging Language 분석
# Loughran-McDonald 재무 어휘 기반 lexicon 정의:
# UNCERTAINTY_WORDS(16개)
#   리스크·불확실성 표현: uncertain, volatile, headwind, risk, concern 등
# HEDGE_WORDS(17개)
#   추정·완화 표현: may, might, approximately, believe, likely 등
UNCERTAINTY_WORDS = {
    'uncertain', 'uncertainty', 'unclear', 'unpredictable', 'volatile',
    'challenging', 'difficult', 'headwind', 'headwinds', 'pressure',
    'risk', 'risks', 'concern', 'concerns', 'cautious', 'caution'
}

HEDGE_WORDS = {
    'may', 'might', 'could', 'should', 'approximately', 'roughly',
    'around', 'about', 'expect', 'expects', 'anticipated', 'believe',
    'believes', 'likely', 'unlikely', 'potentially', 'perhaps'
}

# lexicon_density(): 해당 단어 수 / 전체 단어 수
#   → 문서 길이 차이를 정규화해 기업/분기 간 비교
def lexicon_density(text: str, lexicon: set) -> float:
    if not text or pd.isna(text):
        return np.nan
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return 0.0
    count = sum(1 for w in words if w in lexicon)
    return count / len(words)

df_sample['uncertainty_density'] = df_sample['prepared_remarks'].apply(
    lambda x: lexicon_density(x, UNCERTAINTY_WORDS)
)
df_sample['hedge_density'] = df_sample['prepared_remarks'].apply(
    lambda x: lexicon_density(x, HEDGE_WORDS)
)

print("=== Lexicon Density ===")
print(df_sample[['uncertainty_density', 'hedge_density']].describe().round(5))

# FinBERT neutral score vs Uncertainty density 상관관계
# r > 0, p < 0.05: 모델이 '금융 헤징 언어 = neutral'을 올바르게 포착
# 산점도 그래프
#   좌: uncertainty density, 우: hedge density
valid = df_sample.dropna(subset=['prep_neu_mean', 'uncertainty_density', 'hedge_density'])

r_unc, p_unc = pearsonr(valid['prep_neu_mean'], valid['uncertainty_density'])
r_hdg, p_hdg = pearsonr(valid['prep_neu_mean'], valid['hedge_density'])

print(f"FinBERT neutral ↔ uncertainty_density: r={r_unc:.3f}, p={p_unc:.4g}")
print(f"FinBERT neutral ↔ hedge_density:       r={r_hdg:.3f}, p={p_hdg:.4g}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, label, r, p in [
    (axes[0], 'uncertainty_density', 'Uncertainty Density', r_unc, p_unc),
    (axes[1], 'hedge_density',       'Hedge Density',       r_hdg, p_hdg)
]:
    ax.scatter(valid[col], valid['prep_neu_mean'], alpha=0.5, s=25, color='steelblue')
    m, b = np.polyfit(valid[col], valid['prep_neu_mean'], 1)
    x_line = np.linspace(valid[col].min(), valid[col].max(), 100)
    ax.plot(x_line, m * x_line + b, color='red', linewidth=1.5)
    ax.set_xlabel(label)
    ax.set_ylabel('FinBERT Neutral Mean')
    ax.set_title(f'r={r:.3f}, p={p:.4g}')

plt.suptitle('FinBERT Neutral Score vs Lexicon Density', y=1.02)
plt.tight_layout()
plt.show()

# TF-IDF 기반 감성 극단 콜 특징 단어 추출
#   prep_net_score 상위 25% = 'Positive group'
#   prep_net_score 하위 25% = 'Negative group'
#   두 그룹의 TF-IDF 평균 차이 상위/하위 단어 추출

# TF-IDF 파라미터:
#   min_df=2      : 최소 2개 문서에 등장한 단어만 포함 (노이즈 제거)
#   max_df=0.8    : 80% 이상 문서에 등장하는 범용 단어 제외
#   ngram_range=(1,2): 단어 + 2-gram (e.g., 'strong growth', 'supply chain')
#   양수 = Positive 그룹에서 특징적인 표현 (e.g. 'record revenue')
#   음수 = Negative 그룹에서 특징적인 표현 (e.g. 'supply chain', 'headwinds')
q75 = df_sample['prep_net_score'].quantile(0.75)
q25 = df_sample['prep_net_score'].quantile(0.25)

pos_group = df_sample[df_sample['prep_net_score'] >= q75]['prepared_remarks'].dropna()
neg_group = df_sample[df_sample['prep_net_score'] <= q25]['prepared_remarks'].dropna()

print(f"Positive group (top 25%):    {len(pos_group)} transcripts")
print(f"Negative group (bottom 25%): {len(neg_group)} transcripts")

vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)

all_texts = list(pos_group) + list(neg_group)
labels    = ['pos'] * len(pos_group) + ['neg'] * len(neg_group)

tfidf_mat = vectorizer.fit_transform(all_texts)
vocab     = vectorizer.get_feature_names_out()

tfidf_df  = pd.DataFrame(tfidf_mat.toarray(), columns=vocab)
tfidf_df['label'] = labels

pos_mean_tfidf = tfidf_df[tfidf_df['label'] == 'pos'].drop('label', axis=1).mean()
neg_mean_tfidf = tfidf_df[tfidf_df['label'] == 'neg'].drop('label', axis=1).mean()

diff = (pos_mean_tfidf - neg_mean_tfidf).sort_values(ascending=False)

print("\n=== Positive 그룹 특징 단어 (TF-IDF 차이 상위 15) ===")
print(diff.head(15).round(5))

print("\n=== Negative 그룹 특징 단어 (TF-IDF 차이 하위 15) ===")
print(diff.tail(15).round(5))

# 특징 단어 수평 막대 시각화
#   Positive 특징어(상위 12개): 초록 막대 (diff > 0)
#   Negative 특징어(하위 12개): 빨간 막대 (diff < 0)
#   X축: TF-IDF 평균 차이값, Y축: 단어/2-gram
top_pos = diff.head(12)
top_neg = diff.tail(12)
combined = pd.concat([top_pos, top_neg]).sort_values()

fig, ax = plt.subplots(figsize=(9, 7))
colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in combined.values]
ax.barh(combined.index, combined.values, color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('TF-IDF Mean Difference (Positive - Negative group)')
ax.set_title('Distinguishing Terms: Positive vs Negative Earnings Calls')
plt.tight_layout()
plt.show()


# 분기별 / 기업별 Sentiment 트렌드 분석
# 그래프: 에러바 라인 차트
#   에러바: 표준오차 (std / √n) — n이 적은 분기일수록 에러바 큼
#   회색 점선(y=0): 중립 감성 기준선
#   필터: n ≥ 3인 분기만 시각화 (샘플 수 부족 분기는 대표성 없음)
quarterly = (
    df_sample
    .dropna(subset=['prep_net_score', 'year_q'])
    .groupby('year_q')['prep_net_score']
    .agg(['mean', 'std', 'count'])
    .reset_index()
    .sort_values('year_q')
)

quarterly['se'] = quarterly['std'] / np.sqrt(quarterly['count'])
quarterly = quarterly[quarterly['count'] >= 3]

fig, ax = plt.subplots(figsize=(11, 4))
ax.errorbar(
    quarterly['year_q'], quarterly['mean'],
    yerr=quarterly['se'],
    marker='o', linewidth=1.5, capsize=3, color='steelblue'
)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xticks(range(len(quarterly)))
ax.set_xticklabels(quarterly['year_q'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean Net Sentiment Score')
ax.set_title('Quarterly Earnings Call Sentiment Trend\n(error bar = 1 SE, sample n=300)')
plt.tight_layout()
plt.show()

print(quarterly[['year_q', 'mean', 'count']].to_string(index=False))

# 기업별 분석: 5개 이상 transcript 보유 기업
# 5콜 이상 보유 기업만 선정하여 분기 데이터 기반 기업 고유 성향/실제 변화 분석
#   mean_net: 기업별 평균 감성 (높을수록 일관되게 긍정적 커뮤니케이션)
#   mean_gap: 기업별 평균 tone_gap (클수록 발표↔Q&A 불일치 심한 기업)
#   mean_vol: 기업별 평균 sentiment_volatility (높을수록 어조 불안정)
#   n_calls : 샘플 내 해당 기업 콜 수
ticker_cnt    = df['ticker'].value_counts()
active_tickers = ticker_cnt[ticker_cnt >= 5].index
print(f"5개 이상 transcript 보유 기업 수: {len(active_tickers)}")

df_active = df_sample[df_sample['ticker'].isin(active_tickers)].copy()

ticker_sent = (
    df_active
    .groupby('ticker')
    .agg(
        mean_net=('prep_net_score', 'mean'),
        mean_gap=('tone_gap',       'mean'),
        mean_vol=('sentiment_volatility', 'mean'),
        n_calls =('ticker',         'count')
    )
    .reset_index()
    .sort_values('mean_net', ascending=False)
)

print("\n=== 기업별 Sentiment 상위 10 ===")
print(ticker_sent.head(10).round(4).to_string(index=False))
print("\n=== 기업별 Sentiment 하위 10 ===")
print(ticker_sent.tail(10).round(4).to_string(index=False))


# FinBERT 결과 해석 및 인사이트 요약
# Tone Gap 극단 케이스 분석
# tone_gap 상위 5개:
#   prep_net 높음 + qa_net 낮음 = 공식 발표는 낙관적이었으나 Q&A는 부정적
# tone_gap 하위 5개:
#   prep_net 낮음 + qa_net 높음 = 공식 발표는 부정적이었으나 Q&A는 낙관적
extreme_gap = df_sample.dropna(subset=['tone_gap']).nlargest(5, 'tone_gap')
print("=== Tone Gap 상위 5 (경영진 발표 >> Q&A) ===")
print(extreme_gap[[
    'ticker', 'year_q',
    'prep_net_score', 'qa_net_score', 'tone_gap',
    'prep_pos_mean', 'prep_neg_mean',
    'qa_pos_mean', 'qa_neg_mean'
]].round(4).to_string(index=False))

print("\n=== Tone Gap 하위 5 (Q&A >> 경영진 발표) ===")
extreme_gap2 = df_sample.dropna(subset=['tone_gap']).nsmallest(5, 'tone_gap')
print(extreme_gap2[[
    'ticker', 'year_q',
    'prep_net_score', 'qa_net_score', 'tone_gap'
]].round(4).to_string(index=False))

# Sentiment Volatility vs Uncertainty Density 상관관계
# 두 지표가 양의 상관 → FinBERT가 불확실성 어휘에 반응해 감성 변동성을 높임
valid = df_sample.dropna(subset=['sentiment_volatility', 'uncertainty_density'])
r_vol, p_vol = pearsonr(valid['sentiment_volatility'], valid['uncertainty_density'])
print(f"Sentiment Volatility ↔ Uncertainty Density: r={r_vol:.3f}, p={p_vol:.4g}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(valid['uncertainty_density'], valid['sentiment_volatility'],
           alpha=0.5, s=25, color='#8e44ad')
m, b = np.polyfit(valid['uncertainty_density'], valid['sentiment_volatility'], 1)
x_line = np.linspace(valid['uncertainty_density'].min(), valid['uncertainty_density'].max(), 100)
ax.plot(x_line, m * x_line + b, color='red', linewidth=1.5,
        label=f'r={r_vol:.3f}, p={p_vol:.4g}')
ax.set_xlabel('Uncertainty Word Density (Prepared Remarks)')
ax.set_ylabel('FinBERT Sentiment Volatility (Std of sentence scores)')
ax.set_title('Sentiment Volatility vs Uncertainty Language')
ax.legend()
plt.tight_layout()
plt.show()

# 분석 결과 종합 요약
# 1. 전반적 Sentiment: prep vs qa net_score 평균 비교
# 2. Tone Gap one-sample t-test
#     H0: tone_gap 평균 = 0 (발표와 Q&A 감성 차이 없음)
# 3. Uncertainty 지표: uncertainty_density / hedge_density 평균 (발표 불확실성 기준선)
# 4. 고 Negativity 콜: Q&A neg_ratio > 15%인 콜 수/비율 → 어려운 질문이 집중된 콜 빈도
print("=" * 60)
print("📊 분석 요약 (n=300 sample)")
print("=" * 60)

print(f"\n[1] 전반적 Sentiment")
print(f"  Prepared Net Score 평균: {df_sample['prep_net_score'].mean():.4f}")
print(f"  Q&A Net Score 평균:       {df_sample['qa_net_score'].mean():.4f}")
print(f"  → 경영진 발표가 Q&A보다 {'긍정적' if df_sample['prep_net_score'].mean() > df_sample['qa_net_score'].mean() else '부정적'}")

print(f"\n[2] Tone Gap")
print(f"  평균 Tone Gap: {df_sample['tone_gap'].mean():.4f}")
t_stat, p_val = stats.ttest_1samp(df_sample['tone_gap'].dropna(), popmean=0)
sig = 'p<0.05 → 통계적으로 유의' if p_val < 0.05 else 'p≥0.05 → 통계적으로 유의하지 않음'
print(f"  One-sample t-test: t={t_stat:.3f}, p={p_val:.4g} ({sig})")

print(f"\n[3] Uncertainty")
print(f"  Uncertainty Density 평균: {df_sample['uncertainty_density'].mean():.5f}")
print(f"  Hedge Density 평균:       {df_sample['hedge_density'].mean():.5f}")
print(f"  FinBERT Neutral Mean 평균: {df_sample['prep_neu_mean'].mean():.4f}")

print(f"\n[4] 고 Negativity 콜")
n_high = df_sample['high_negativity'].sum()
print(f"  Q&A neg_ratio > 15%: {n_high}개 ({n_high/len(df_sample):.1%})")

print("\n" + "=" * 60)


# 섹터별 Sentiment 시계열 분석
# Ticker → GICS Sector 매핑
# yfinance Ticker.info에서 sector / industry / marketCap 추출
# GICS 기준 = S&P500 분류와 동일한 금융권 표준
# dict 캐싱: 중복 API 호출 방지
# time.sleep(0.05): rate limit 방지 (초당 20건 이하 유지)
def fetch_sector_info(ticker: str, cache: dict) -> dict:
    if ticker in cache:
        return cache[ticker]
    try:
        info   = yf.Ticker(ticker).info
        result = {
            'sector'   : info.get('sector',    None),
            'industry' : info.get('industry',  None),
            'marketCap': info.get('marketCap', None),
        }
    except Exception:
        result = {'sector': None, 'industry': None, 'marketCap': None}
    cache[ticker] = result
    return result

unique_tickers = df['ticker'].dropna().unique()
print(f"매핑할 고유 티커 수: {len(unique_tickers)}")

sector_cache = {}
rows = []
for tkr in tqdm(unique_tickers, desc="Fetching sector info"):
    info = fetch_sector_info(tkr, sector_cache)
    rows.append({'ticker': tkr, **info})
    time.sleep(0.05)

sector_map = pd.DataFrame(rows)
print("\n섹터 분포:")
print(sector_map['sector'].value_counts())

# 섹터별 층화 샘플링
# 각 섹터 비율에 맞게 추출하되 그룹당 최소 5건 보장
N_SECTOR_SAMPLE = 1000 # N_SECTOR_SAMPLE: 조절 가능 

df_with_sector = df.merge(sector_map, on='ticker', how='inner')
df_with_sector = df_with_sector[df_with_sector['sector'].notna()].copy()
print(f"섹터 정보 있는 transcript: {len(df_with_sector):,}건")

def stratified_sample(df, group_col, n_total, min_per_group=5):
    """group_col 기준 층화 샘플링. 비율 기준 n_total건 추출, 그룹당 최소 min_per_group건 보장."""
    counts = df[group_col].value_counts(normalize=True)
    result = []
    for grp, prop in counts.items():
        n_grp  = max(min_per_group, int(n_total * prop))
        subset = df[df[group_col] == grp]
        result.append(subset.sample(min(n_grp, len(subset)), random_state=42))
    return pd.concat(result).reset_index(drop=True)

df_sector = stratified_sample(df_with_sector, 'sector', N_SECTOR_SAMPLE)
print(f"\n층화 샘플링 후 섹터별 건수:")
print(df_sector['sector'].value_counts())

# FinBERT 적용 (Prepared Remarks 기준)
# 섹터 간 비교 시 더 일관된 언어 구조로 비교 안전성이 더 높은 prepared 섹션 사용
tqdm.pandas(desc="Sector FinBERT (Prepared)")
sector_results = df_sector['prepared_remarks'].progress_apply(sentences_to_finbert)

sector_feat         = pd.json_normalize(sector_results)
sector_feat.columns = [f"prep_{c}" for c in sector_feat.columns]
df_sector           = pd.concat([df_sector.reset_index(drop=True), sector_feat], axis=1)

print("FinBERT 완료. 컬럼:", [c for c in df_sector.columns if c.startswith('prep_')])
print(df_sector[['ticker', 'sector', 'year', 'prep_net_score']].head())

# 섹터 × 연도 집계 (시총 가중 평균 포함)
# mean_net  : 단순 평균 (건수 기준)
# wcap_net  : 시총 가중 평균 — 대형주 영향력 반영
# std_net   : 표준편차 — 섹터 내 기업 간 의견 분산
# marketCap_w: 시총 결측치는 중앙값으로 대체
df_sector['marketCap_w'] = df_sector['marketCap'].fillna(df_sector['marketCap'].median())

def weighted_mean(group):
    """시총 가중 평균 net score."""
    valid = group.dropna(subset=['prep_net_score'])
    if len(valid) == 0:
        return np.nan
    return np.average(valid['prep_net_score'], weights=valid['marketCap_w'])

sector_yearly = (
    df_sector
    .dropna(subset=['year', 'sector', 'prep_net_score'])
    .groupby(['year', 'sector'])
    .agg(
        mean_net=('prep_net_score', 'mean'),
        std_net =('prep_net_score', 'std'),
        n       =('prep_net_score', 'count'),
        mean_pos=('prep_pos_mean',  'mean'),
        mean_neg=('prep_neg_mean',  'mean'),
        mean_neu=('prep_neu_mean',  'mean'),
    )
    .reset_index()
)

_wcap_cols = df_sector.dropna(subset=['year', 'sector', 'prep_net_score'])[
    ['year', 'sector', 'prep_net_score', 'marketCap_w']
]
wcap_avg = (
    _wcap_cols
    .groupby(['year', 'sector'])
    .apply(weighted_mean)
    .reset_index()
    .rename(columns={0: 'wcap_net'})
)
sector_yearly = sector_yearly.merge(wcap_avg, on=['year', 'sector'])

print("=== 섹터-연도 집계 결과 샘플 ===")
print(sector_yearly.sort_values(['sector', 'year']).head(20).round(4).to_string(index=False))

# Z-Score: 이례적으로 좋은/나쁜 시기 포착
# Z-Score = (해당 연도 점수 - 섹터 전체 기간 평균) / 표준편차
#   Z > 1.5  → BOOM: 해당 연도에 유독 긍정적 
#   Z < -1.5 → BUST: 해당 연도에 유독 부정적 
#   그 외    → normal: 통상적인 수준
sector_stats = (
    sector_yearly
    .groupby('sector')['mean_net']
    .agg(['mean', 'std'])
    .rename(columns={'mean': 'all_mean', 'std': 'all_std'})
    .reset_index()
)
sector_yearly = sector_yearly.merge(sector_stats, on='sector')
sector_yearly['z_score'] = (
    (sector_yearly['mean_net'] - sector_yearly['all_mean'])
    / sector_yearly['all_std'].replace(0, np.nan)
)
sector_yearly['anomaly'] = sector_yearly['z_score'].apply(
    lambda z: 'BOOM' if z > 1.5 else ('BUST' if z < -1.5 else 'normal')
)

print("=== Z-Score 이상치 (BOOM/BUST 시기) ===")
anomalies = sector_yearly[sector_yearly['anomaly'] != 'normal'].sort_values('z_score', ascending=False)
print(anomalies[['year', 'sector', 'mean_net', 'z_score', 'anomaly', 'n']].to_string(index=False))

# 시계열 히트맵: 섹터 × 연도
# 그래프: 히트맵 2개 (좌: Net Score, 우: Z-Score)
# 읽는 법:
#   가로 패턴 (같은 연도 모든 행 동시 변색): 시장 전체 충격
#   세로 패턴 (특정 섹터만 지속 어두움): 구조적 쇠퇴 산업
#   Z-Score에서 특정 셀만 극단값: 섹터 특수 이벤트 발생
pivot_net = sector_yearly.pivot_table(index='sector', columns='year', values='mean_net')
pivot_z   = sector_yearly.pivot_table(index='sector', columns='year', values='z_score')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, pivot, title, vmin, vmax in [
    (axes[0], pivot_net, 'Net Score\nred=neg, green=pos', -0.3, 0.6),
    (axes[1], pivot_z,   'Z-Score\n(vs sector baseline)', -2.5, 2.5),
]:
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f'Sector Sentiment {title}')
    plt.colorbar(im, ax=ax)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6)

plt.tight_layout()
plt.show()

# 사이클 / 성장 / 하락 산업 자동 분류
# 선형 회귀 기반 두 지표로 5가지 유형 자동 분류:
# slope (기울기): 추세 방향
#   > +0.02: 성장(Growing)
#   < -0.02: 하락(Declining)
# residual_std (추세선 잔차 표준편차): 사이클성
#   상위 40% → 오르내림 반복 = Cyclical (경기 민감 산업)
#
# 5가지 유형:
#   Growing             : slope 양수, residual_std 낮음
#   Stable              : slope 작음, residual_std 낮음
#   Cyclical            : residual_std 높음, slope 양수 또는 작음
#   Structurally Declining: slope 음수, residual_std 낮음
#   Cyclical & Declining: slope 음수, residual_std 높음
sector_trend = []
for sector, grp in sector_yearly.groupby('sector'):
    grp = grp.dropna(subset=['year', 'mean_net']).sort_values('year').reset_index(drop=True)
    if len(grp) < 2:
        continue
    slope, intercept, r, p, se = linregress(grp['year'], grp['mean_net'])
    fitted       = intercept + slope * grp['year'].values
    residual_std = (grp['mean_net'].values - fitted).std()
    sector_trend.append({
        'sector'       : sector,
        'slope'        : slope,
        'r_squared'    : r**2,
        'p_value'      : p,
        'residual_std' : residual_std,
        'n_years'      : len(grp),
    })

sector_trend_df = pd.DataFrame(sector_trend)

def classify_sector(row):
    high_cycle = row['residual_std'] > sector_trend_df['residual_std'].quantile(0.6)
    declining  = row['slope'] < -0.02
    growing    = row['slope'] >  0.02
    if high_cycle and declining:
        return 'Cyclical & Declining'
    elif high_cycle:
        return 'Cyclical'
    elif declining:
        return 'Structurally Declining'
    elif growing:
        return 'Growing'
    else:
        return 'Stable'

sector_trend_df['type'] = sector_trend_df.apply(classify_sector, axis=1)
print("=== 산업 유형 분류 ===")
print(sector_trend_df[['sector', 'slope', 'residual_std', 'r_squared', 'p_value', 'type']]
      .sort_values('slope').round(4).to_string(index=False))

# 섹터별 sentiment 시계열 라인 차트
# 그래프: 멀티 라인 차트
#   선 색상: 유형별 구분
#     Growing=초록, Stable=파랑, Cyclical=주황,
#     Structurally Declining=빨강, Cyclical & Declining=진빨강
#   선 스타일: Growing/Stable=실선, Cyclical=점선,
#              Declining=점-실선, Cyclical&Declining=짧은점선
#   음영 영역: ±1 std (섹터 내 기업 간 분산, n≥3 구간에만 표시)
#   필터: 전체 기간 합산 n ≥ 5인 섹터만 표시
TYPE_COLORS = {
    'Growing'               : '#2ecc71',
    'Stable'                : '#3498db',
    'Cyclical'              : '#f39c12',
    'Structurally Declining': '#e74c3c',
    'Cyclical & Declining'  : '#c0392b',
}
TYPE_STYLE = {
    'Growing'               : '-',
    'Stable'                : '-',
    'Cyclical'              : '--',
    'Structurally Declining': '-.',
    'Cyclical & Declining'  : ':',
}

sector_yearly_typed = sector_yearly.merge(
    sector_trend_df[['sector', 'type']], on='sector', how='left'
)
valid_sectors = sector_yearly.groupby('sector')['n'].sum()
valid_sectors = valid_sectors[valid_sectors >= 5].index

fig, ax = plt.subplots(figsize=(13, 6))
for sector in valid_sectors:
    grp    = sector_yearly_typed[sector_yearly_typed['sector'] == sector].sort_values('year')
    s_type = grp['type'].iloc[0] if len(grp) > 0 else 'Stable'
    color  = TYPE_COLORS.get(s_type, 'gray')
    style  = TYPE_STYLE.get(s_type, '-')

    ax.plot(grp['year'], grp['mean_net'],
            marker='o', markersize=4, linewidth=1.5,
            color=color, linestyle=style, label=f"{sector} ({s_type})")

    grp_v = grp[grp['n'] >= 3]
    if len(grp_v) > 0:
        ax.fill_between(grp_v['year'],
                        grp_v['mean_net'] - grp_v['std_net'],
                        grp_v['mean_net'] + grp_v['std_net'],
                        alpha=0.08, color=color)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Year')
ax.set_ylabel('Mean Net Sentiment Score')
ax.set_title('Sector Sentiment Trend by Year\n(shaded area = ±1 std within sector)')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)
plt.tight_layout()
plt.show()

# 연도별 Top-K 기업 → 지배 산업 추출
# 연도별 net_score 상위 20%(TOP_K_PERCENTILE) 기업 선정 후 섹터 빈도 집계
# 두 가지 관점으로 비교:
#   freq     (건수 기준): 단순 출현 빈도 — 작은 기업도 동등하게 반영
#   wcap_freq(시총 기준): 대형주 영향력 반영 — 시장 관점의 주도 섹터
TOP_K_PERCENTILE = 0.80

dominant_rows = []
for year, grp in df_sector.dropna(subset=['prep_net_score', 'sector', 'year']).groupby('year'):
    threshold = grp['prep_net_score'].quantile(TOP_K_PERCENTILE)
    top_k     = grp[grp['prep_net_score'] >= threshold]
    freq      = top_k['sector'].value_counts(normalize=True)
    wcap_freq = (
        top_k.groupby('sector')['marketCap_w'].sum() / top_k['marketCap_w'].sum()
    ).sort_values(ascending=False)

    dominant_rows.append({
        'year'           : year,
        'top1_sector'    : freq.index[0]      if len(freq) > 0 else None,
        'top1_freq'      : freq.iloc[0]       if len(freq) > 0 else None,
        'top2_sector'    : freq.index[1]      if len(freq) > 1 else None,
        'top1_wcap'      : wcap_freq.index[0] if len(wcap_freq) > 0 else None,
        'top1_wcap_share': wcap_freq.iloc[0]  if len(wcap_freq) > 0 else None,
        'n_top_companies': len(top_k),
    })

dominant_df = pd.DataFrame(dominant_rows).sort_values('year')
print("=== 연도별 주도 산업 (Top 20% 기업 기준) ===")
print(dominant_df.round(3).to_string(index=False))

# 키워드 드리프트 (같은 산업인데 시기별로 핵심 언어가 어떻게 바뀌는가)
#   사이클성(residual_std)이 가장 높은 섹터 자동 선정
#   연도별 transcript를 합쳐서 TF-IDF → 연도별 Top-10 키워드 추출
target_sector = sector_trend_df.sort_values('residual_std', ascending=False)['sector'].iloc[0]
print(f"\n키워드 드리프트 분석 대상 섹터: [{target_sector}]")
print("(사이클성이 가장 높은 섹터 = 언어 변화도 클 것으로 예상)\n")

sector_texts = (
    df_sector[df_sector['sector'] == target_sector]
    .dropna(subset=['year', 'prepared_remarks'])
    .groupby('year')['prepared_remarks']
    .apply(lambda x: ' '.join(x))
)

if len(sector_texts) >= 2:
    vect      = TfidfVectorizer(max_features=500, stop_words='english',
                                ngram_range=(1, 2), min_df=1)
    tfidf_mat = vect.fit_transform(sector_texts.values)
    vocab     = vect.get_feature_names_out()
    tfidf_df  = pd.DataFrame(tfidf_mat.toarray(), index=sector_texts.index, columns=vocab)

    print(f"{'연도':<8} {'Top-10 키워드'}")
    print("-" * 60)
    for year in tfidf_df.index:
        top_words = tfidf_df.loc[year].nlargest(10).index.tolist()
        print(f"{int(year):<8} {', '.join(top_words)}")
else:
    print("해당 섹터의 연도별 데이터 부족. target_sector를 직접 지정해주세요.")
    print("ex) target_sector = 'Technology'")

# 최종 리포트: 연도별 섹터 Sentiment 랭킹
# 연도별로 섹터 감성 점수 내림차순 정렬
# 출력 컬럼: Rank / Sector / Net Score / Z-Score / N / Anomaly 태그
#   BOOM: Z > 1.5 (평소 대비 유독 긍정적인 해)
#   BUST: Z < -1.5 (평소 대비 유독 부정적인 해)
print("=" * 70)
print(" 연도별 섹터 Sentiment 랭킹 리포트")
print("   (mean_net: 높을수록 긍정 | z_score: 평소 대비 이례적 수준)")
print("=" * 70)

for year in sorted(sector_yearly['year'].dropna().unique()):
    yr_data = (
        sector_yearly[sector_yearly['year'] == year]
        .dropna(subset=['mean_net'])
        .sort_values('mean_net', ascending=False)
        .reset_index(drop=True)
    )
    if len(yr_data) == 0:
        continue
    print(f"\n[{int(year)}년]")
    print(f"  {'Rank':<5} {'Sector':<30} {'Net Score':>10} {'Z-Score':>9} {'N':>5} {'Anomaly':>8}")
    print("  " + "-" * 65)
    for rank, row in yr_data.iterrows():
        tag = f"🔥BOOM" if row['anomaly'] == 'BOOM' else ("❄️BUST" if row['anomaly'] == 'BUST' else '')
        z   = row['z_score'] if not pd.isna(row.get('z_score', np.nan)) else float('nan')
        print(f"  {rank+1:<5} {row['sector']:<30} {row['mean_net']:>10.4f} "
              f"{z:>9.2f} {int(row['n']):>5}  {tag}")
