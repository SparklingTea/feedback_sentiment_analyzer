import json
import random

import anthropic

from scraper_agent import FALLBACK_BETA, MODEL, _secret

MAX_PER_SENTIMENT = 300

THEME_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string", "enum": ["Positive", "Neutral", "Negative"]},
                    "themes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "theme": {"type": "string"},
                                "kind": {"type": "string", "enum": ["feature", "emotion", "other"]},
                                "comment_ids": {"type": "array", "items": {"type": "integer"}},
                                "quote_id": {"type": "integer"},
                            },
                            "required": ["theme", "kind", "comment_ids", "quote_id"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["sentiment", "themes"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["groups"],
    "additionalProperties": False,
}

PROMPT = """These are customer reviews and social media comments about a product, already grouped by sentiment.
For each sentiment group, find up to 6 themes that explain WHY people feel that way.

Good themes are specific and informative:
- product features or attributes people praise or criticise ("autofocus speed", "battery life",
  "price vs value", "easy for beginners", "build quality", "delivery and packaging")
- emotions or experiences ("excited to upgrade", "regret after buying", "comparing with Sony")
Bad themes are single generic words ("good", "use", "camera") or chat filler. For Neutral, questions and
comparisons are useful themes; skip comments that say nothing about the product.

Rules:
- Name themes in plain English, 2-5 words, even if a comment is in another language.
- comment_ids: every comment in that group that clearly expresses the theme. Only use ids from that group.
- Only keep a theme if at least 2 comments support it (1 if the group has fewer than 10 comments).
- quote_id: the one comment from comment_ids that best illustrates the theme.
- Order themes from most to least supported.

{groups}"""


def sentiment_themes(df, text_col, sentiment_col='Sentiment'):
    """Returns {sentiment: {"n": reviews analysed, "sampled": bool,
    "themes": [{"theme", "kind", "count", "share", "quote"}]}}.
    Counts and quotes come from the comment ids the model assigns, not from the model's own wording."""
    rng = random.Random(0)
    comments, groups, meta = {}, [], {}
    for sentiment in ("Positive", "Neutral", "Negative"):
        texts = [t for t in df.loc[df[sentiment_col] == sentiment, text_col].dropna().astype(str) if t.strip()]
        sampled = len(texts) > MAX_PER_SENTIMENT
        if sampled:
            texts = rng.sample(texts, MAX_PER_SENTIMENT)
        meta[sentiment] = {"n": len(texts), "sampled": sampled, "themes": []}
        if not texts:
            continue
        lines = []
        for t in texts:
            cid = len(comments)
            comments[cid] = (sentiment, t)
            lines.append(f"[{cid}] {' '.join(t.split())[:400]}")
        groups.append(f"=== {sentiment} ({len(texts)} comments) ===\n" + "\n".join(lines))
    if not groups:
        return meta

    client = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))
    response = client.beta.messages.create(
        model=MODEL, max_tokens=16000,
        messages=[{"role": "user", "content": PROMPT.format(groups="\n\n".join(groups))}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": THEME_SCHEMA}},
        betas=[FALLBACK_BETA], fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The AI model declined to summarise these reviews.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The theme summary was cut off.")
    result = json.loads(next(b.text for b in response.content if b.type == "text"))

    for group in result["groups"]:
        sentiment = group["sentiment"]
        for theme in group["themes"]:
            ids = sorted({i for i in theme["comment_ids"] if comments.get(i, (None,))[0] == sentiment})
            if not ids:
                continue
            quote_id = theme["quote_id"] if theme["quote_id"] in ids else ids[0]
            quote = " ".join(comments[quote_id][1].split())
            meta[sentiment]["themes"].append({
                "theme": theme["theme"], "kind": theme["kind"], "count": len(ids),
                "share": len(ids) / meta[sentiment]["n"],
                "quote": quote if len(quote) <= 180 else quote[:177].rsplit(" ", 1)[0] + "…",
            })
        meta[sentiment]["themes"].sort(key=lambda t: -t["count"])
    return meta
