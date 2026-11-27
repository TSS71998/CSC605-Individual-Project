"""LLM assistant integrated through LangChain (Gemini or OpenAI API).

How it is used in this project
------------------------------
* Each major data-preparation / analysis decision has a prompt template (DECISIONS).
  The prompt is filled with a SUMMARY computed from the data (never raw rows).
* Every call is appended to ai_log/interactions.jsonl (model, full prompt,
  full response), so the report shows the real interactions, even without an API key.
* After reading a response, the verdict (Accepted / Modified /
  Rejected) and a justification are recorded in ai_log/decisions.json; the report displays them.
"""
import json
import os
from datetime import datetime, timezone

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import AI_LOG, ROOT

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

INTERACTIONS = AI_LOG / "interactions.jsonl"
VERDICTS = AI_LOG / "decisions.json"

DEFAULT_MODELS = {"google": "gemini-3.5-flash", "openai": "gpt-4.1-mini"}
KEY_VARS = {"google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}

def settings():
    provider = os.getenv("LLM_PROVIDER", "google").strip().lower()
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider, "")
    key_var = KEY_VARS.get(provider, "")
    return {"provider": provider, "model": model, "key_var": key_var,
            "has_key": bool(key_var and os.getenv(key_var))}

def get_llm(temperature=0.2):
    s = settings()
    if s["provider"] == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=s["model"], temperature=temperature)
    if s["provider"] == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=s["model"], temperature=temperature)
    raise ValueError(f"Unknown LLM_PROVIDER '{s['provider']}' (use 'google' or 'openai').")

SYSTEM_STRUCTURED = """You are a senior data scientist reviewing a student's music-genre analysis \
of the FMA (Free Music Archive) dataset. You receive SUMMARY STATISTICS computed from the data, \
never the raw rows; do not invent numbers that are not in the context.

Answer in this exact format (under 250 words total):
**Recommendation:** one or two sentences.
**Reasoning:** 2-4 bullet points that refer to the numbers in the context.
**Assumptions / risks:** 1-3 bullet points (what could make this recommendation wrong).
**Alternative:** one alternative approach and when it would be better.

Describe relationships as associations, not causes."""

SYSTEM_NAIVE = "You are a helpful assistant."

HUMAN_STRUCTURED = """Project goal: predict a track's top-level genre from pre-computed audio \
features and find which genres are confused.

Analysis stage: {stage}

Context (computed from the data):
{context}

Question: {question}"""

HUMAN_NAIVE = "{question}"

def build_chain(structured=True, llm=None):
    """LangChain LCEL pipeline: prompt template -> chat model -> string."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_STRUCTURED if structured else SYSTEM_NAIVE),
        ("human", HUMAN_STRUCTURED if structured else HUMAN_NAIVE),
    ])
    return prompt | (llm or get_llm()) | StrOutputParser()

def render_prompt(structured, **kw):
    """The exact text sent to the model (for documentation in the report)."""
    system = SYSTEM_STRUCTURED if structured else SYSTEM_NAIVE
    human = (HUMAN_STRUCTURED if structured else HUMAN_NAIVE).format(**kw)
    return f"[SYSTEM]\n{system}\n\n[USER]\n{human}"

def ask(question, context="", stage="General", decision_id="free", structured=True, llm=None):
    """Call the model through LangChain and log the interaction. Returns the response text."""
    chain = build_chain(structured, llm)
    inputs = {"question": question, "context": context, "stage": stage}
    if not structured:
        inputs = {"question": question}
    response = chain.invoke(inputs)
    s = settings()
    log_interaction({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision_id": decision_id, "variant": "structured" if structured else "naive",
        "provider": s["provider"], "model": s["model"], "stage": stage, "question": question,
        "prompt": render_prompt(structured, **({"question": question, "context": context, "stage": stage}
                                               if structured else {"question": question})),
        "response": response,
    })
    return response

def log_interaction(record):
    AI_LOG.mkdir(parents=True, exist_ok=True)
    with open(INTERACTIONS, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_interactions():
    if not INTERACTIONS.exists():
        return []
    return [json.loads(line) for line in INTERACTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]

def latest(decision_id, variant="structured"):
    hits = [r for r in load_interactions() if r["decision_id"] == decision_id and r.get("variant") == variant]
    return hits[-1] if hits else None

def load_verdicts():
    if not VERDICTS.exists():
        return {}
    return json.loads(VERDICTS.read_text(encoding="utf-8"))

DECISIONS = [
    {
        "id": "header_parsing", "stage": "Data cleaning: loading", "context": "headers",
        "title": "Reading the multi-row CSV headers",
        "question": "tracks.csv has a 2-row header and features.csv a 3-row header. How should I load "
                    "them in pandas so the columns are usable, and what pitfalls should I watch for?",
        "implemented": "pd.read_csv(header=[0, 1]) and header=[0, 1, 2] with index_col=0, then flatten the "
                       "MultiIndex to names like mfcc_mean_01; dates parsed with errors='coerce'."
    },
    {
        "id": "missing_genre", "stage": "Data cleaning: missing values", "context": "missing",
        "title": "Handling the missing genre label",
        "question": "genre_top (my target label) is missing for a large share of tracks, and several "
                    "metadata columns are mostly empty. How should I handle these missing values?",
        "implemented": "Rows without genre_top are kept for general EDA but excluded from genre-based tests "
                       "and models; the label is not imputed. Mostly-empty free-text columns are not "
                       "carried forward. Missing release years are kept as 'Unknown'."
    },
    {
        "id": "duration_outliers", "stage": "Data cleaning: outliers", "context": "duration",
        "title": "Duration outliers",
        "naive_question": "how do I remove outliers from track duration",
        "question": "Track duration is right-skewed with some zero-length tracks and some tracks over an "
                    "hour long. Which values should I treat as outliers, and should I remove, cap or keep them?",
        "implemented": "Duration = 0 set to missing (impossible). IQR rule applied on log(duration); flagged "
                       "tracks are kept (long DJ mixes / live sets are real)."
    },
    {
        "id": "listens_skew", "stage": "EDA: transformations", "context": "listens",
        "title": "Skewed listen counts",
        "question": "Listens are extremely right-skewed. How should I treat them for EDA and for the "
                    "correlation test with audio features?",
        "implemented": "log1p transform; all tracks kept; Spearman correlation (rank-based) for the test."
    },
    {
        "id": "test_choice", "stage": "Statistical analysis", "context": "anova",
        "title": "Choosing the test for energy across genres",
        "question": "I want to test whether mean energy differs across genres. Given these assumption "
                    "checks, which test should be my primary test and how should I report effect size "
                    "and post-hoc comparisons?",
        "implemented": "Before this review the rule picked Kruskal-Wallis whenever residuals were non-normal, "
                       "with Tukey HSD post-hoc tests. After it: Welch's ANOVA is primary when variances differ "
                       "and every genre has >= 30 tracks, with Games-Howell post-hoc tests and eta²/omega²; "
                       "Kruskal-Wallis with Holm-corrected Mann-Whitney tests is kept as a robustness check."
    },
    {
        "id": "model_choice", "stage": "Machine learning: model and metric choice", "context": "models",
        "title": "Models, metrics and class balance",
        "question": "Which baseline and advanced models should I use to classify the 8 genres from these "
                    "518 features, which evaluation metrics should I report, and do I need to "
                    "resample (e.g. SMOTE) for class imbalance?",
        "implemented": "Logistic Regression (baseline), k-NN (reference), Random Forest and HistGradientBoosting "
                       "(advanced); tuned on the official validation split by macro-F1; no resampling."
    },
    {
        "id": "interpret_results", "stage": "Model evaluation: interpretation", "context": "results",
        "title": "Interpreting the confusion matrix and feature importance",
        "question": "Interpret these model results for a non-technical audience. Which genres are "
                    "confused and why might that be? What do the feature importances suggest?",
        "implemented": "Conclusions section: report best-model metrics vs. chance and baseline, the top "
                       "confused pairs, and the most important feature families, all as associations."
    },
]

PROMPT_DEMO_ID = "duration_outliers"  # decision used for the naive vs structured prompt comparison