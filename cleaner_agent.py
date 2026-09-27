import json

import anthropic
import pandas as pd

from scraper_agent import FALLBACK_BETA, MODEL, _secret

MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "text_column": {"type": "string"},
        "title_column": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "date_column": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "rating_column": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "rating_max": {"anyOf": [{"type": "number"}, {"type": "null"}]},
    },
    "required": ["text_column", "title_column", "date_column", "rating_column", "rating_max"],
    "additionalProperties": False,
}

PROMPT = """Below are the columns of a table of scraped {platform} customer feedback, with sample values.
Identify which column holds:
- text_column: the review/comment body written by the customer (not the product description, not a URL)
- title_column: a separate short review headline, if there is one, else null
- date_column: when the review/comment was posted (not when it was scraped), else null
- rating_column: the customer's star or numeric rating of the product, else null. Not helpful-vote counts or likes.
- rating_max: the top of that rating scale (e.g. 5), else null
Use exact column names from the list.

{samples}"""


def _describe_columns(df, n_samples=5):
    lines = []
    for col in df.columns:
        values = df[col].dropna().astype(str).head(n_samples).tolist()
        lines.append(f"{col}: {json.dumps([v[:150] for v in values])}")
    return "\n".join(lines)


def infer_mapping(df, platform, client):
    """Only the column names and a few sample values go to the model, never the full table."""
    response = client.beta.messages.create(
        model=MODEL, max_tokens=2000,
        messages=[{"role": "user", "content": PROMPT.format(platform=platform, samples=_describe_columns(df))}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": MAPPING_SCHEMA}},
        betas=[FALLBACK_BETA], fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"The AI model declined to map the {platform} columns.")
    mapping = json.loads(next(b.text for b in response.content if b.type == "text"))

    if mapping["text_column"] not in df.columns:
        raise RuntimeError(f"{platform}: model picked unknown text column '{mapping['text_column']}'")
    for key in ("title_column", "date_column", "rating_column"):
        if mapping[key] not in df.columns:
            mapping[key] = None
    return mapping


def _to_datetime(col):
    numeric = pd.to_numeric(col, errors="coerce")
    if numeric.notna().mean() > 0.9:
        # Unix timestamps (TikTok's createTime) - pandas would otherwise read them as nanoseconds
        return pd.to_datetime(numeric, unit="s" if numeric.max() < 1e11 else "ms", errors="coerce", utc=True)
    return pd.to_datetime(col, errors="coerce", utc=True, format="mixed")


def apply_mapping(df, mapping, platform):
    """Keeps every row that has feedback text; only the columns are reduced."""
    text = df[mapping["text_column"]].astype("string").str.strip()
    if mapping["title_column"]:
        title = df[mapping["title_column"]].astype("string").str.strip().fillna("")
        title = title.where(title.str.contains(r"[.!?]$") | (title == ""), title + ".")
        text = (title + " " + text.fillna("")).str.strip().where(text.notna() | (title != ""))

    out = pd.DataFrame({"platform": platform, "text": text})
    out["date"] = _to_datetime(df[mapping["date_column"]]) if mapping["date_column"] else pd.NaT
    if mapping["rating_column"]:
        rating = pd.to_numeric(df[mapping["rating_column"]].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],
                               errors="coerce")
        if mapping["rating_max"] and mapping["rating_max"] != 5:
            rating = rating / mapping["rating_max"] * 5
        out["rating"] = rating
    else:
        out["rating"] = pd.NA
    return out[out["text"].fillna("").str.len() > 0]


def clean_reviews(raw, log=print):
    """raw: {platform: DataFrame}. Returns one table with columns platform, text, date, rating."""
    client = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))
    frames = []
    for platform, df in raw.items():
        if df is None or df.empty:
            continue
        mapping = infer_mapping(df, platform, client)
        log(f"{platform}: text='{mapping['text_column']}', date='{mapping['date_column']}', "
            f"rating='{mapping['rating_column']}'")
        frames.append(apply_mapping(df, mapping, platform))
    if not frames:
        return pd.DataFrame(columns=["platform", "text", "date", "rating"])

    cleaned = pd.concat(frames, ignore_index=True)
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["platform", "text"]).reset_index(drop=True)
    log(f"Kept {len(cleaned)} rows ({before - len(cleaned)} exact duplicates removed)")
    return cleaned
