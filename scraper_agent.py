import time

import anthropic
import pandas as pd
import requests
import streamlit as st

MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
APIFY_BASE = "https://api.apify.com/v2"

ACTORS = {
    "amazon": "junglee~amazon-reviews-scraper",
    "ebay": "web_wanderer~ebay-reviews-scraper",
    "tiktok_search": "clockworks~tiktok-scraper",
    "tiktok_comments": "clockworks~tiktok-comments-scraper",
}

SEARCH_DOMAINS = {
    "Amazon": ["amazon.com", "amazon.co.uk", "amazon.ca", "amazon.de", "amazon.com.au"],
    "eBay": ["ebay.com", "ebay.co.uk", "ebay.ca", "ebay.de", "ebay.com.au"],
}

SYSTEM_PROMPT = """You collect customer feedback about one product so it can be analysed for sentiment.

The user names a product (and sometimes gives links) plus the platforms to collect from.
For each selected platform:
- Amazon / eBay: if no product link was given, use web search to find the product's own listing page
  (an Amazon /dp/ page, an eBay /itm/ page). Pick the listing that best matches the product, preferring
  ones with many reviews. Then call the platform's scrape tool once with 1-3 listing URLs.
- TikTok: if no video links were given, call find_tiktok_videos with a short product search term, choose
  the videos that are actually about the product (reviews, unboxings, comparisons) with the most comments,
  then call scrape_tiktok_comments once with those video URLs.

Scrape each platform once - the tools already collect as many reviews as the user allowed.
If a tool returns an error or zero rows, you may retry that platform once with different URLs.
Finish with two or three sentences saying what was collected from where."""


def _secret(name):
    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError):
        raise RuntimeError(f"Missing '{name}' in Streamlit secrets.")


def _run_actor(actor, run_input, max_charge_usd):
    token = _secret("APIFY_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}
    run = requests.post(
        f"{APIFY_BASE}/acts/{actor}/runs",
        headers=headers,
        params={"waitForFinish": 60, "maxTotalChargeUsd": max_charge_usd},
        json=run_input,
        timeout=90,
    )
    run.raise_for_status()
    run = run.json()["data"]

    deadline = time.time() + 15 * 60
    while run["status"] in ("READY", "RUNNING") and time.time() < deadline:
        run = requests.get(f"{APIFY_BASE}/actor-runs/{run['id']}", headers=headers,
                           params={"waitForFinish": 60}, timeout=90).json()["data"]
    if run["status"] != "SUCCEEDED":
        raise RuntimeError(f"Apify run {run['id']} ended with status {run['status']}")

    items = requests.get(f"{APIFY_BASE}/datasets/{run['defaultDatasetId']}/items",
                         headers=headers, params={"clean": "true", "format": "json"}, timeout=120)
    items.raise_for_status()
    return items.json()


def _tool_defs(platforms):
    urls = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3}
    tools = []
    search_domains = [d for p in platforms for d in SEARCH_DOMAINS.get(p, [])]
    if search_domains:
        tools.append({"type": "web_search_20260209", "name": "web_search",
                      "max_uses": 6, "allowed_domains": search_domains})
    if "Amazon" in platforms:
        tools.append({"name": "scrape_amazon_reviews",
                      "description": "Scrape customer reviews from Amazon product pages (/dp/ URLs).",
                      "input_schema": {"type": "object", "properties": {"product_urls": urls},
                                       "required": ["product_urls"], "additionalProperties": False}})
    if "eBay" in platforms:
        tools.append({"name": "scrape_ebay_reviews",
                      "description": "Scrape buyer reviews and feedback from eBay item pages (/itm/ URLs).",
                      "input_schema": {"type": "object", "properties": {"product_urls": urls},
                                       "required": ["product_urls"], "additionalProperties": False}})
    if "TikTok" in platforms:
        tools.append({"name": "find_tiktok_videos",
                      "description": "Search TikTok videos. Returns URL, caption and comment count per video.",
                      "input_schema": {"type": "object",
                                       "properties": {"query": {"type": "string"}},
                                       "required": ["query"], "additionalProperties": False}})
        tools.append({"name": "scrape_tiktok_comments",
                      "description": "Scrape comments from TikTok video URLs.",
                      "input_schema": {"type": "object",
                                       "properties": {"video_urls": {**urls, "maxItems": 5}},
                                       "required": ["video_urls"], "additionalProperties": False}})
    return tools


def run_scraper_agent(request, platforms, max_reviews, max_charge_usd=5.0, log=print):
    """Let Claude find the product on each platform and scrape it.
    Returns {platform: DataFrame of raw scraped rows} - the raw rows never go to the model."""
    collected = {}

    def store(platform, items):
        if items:
            frame = pd.json_normalize(items)
            collected[platform] = pd.concat([collected.get(platform), frame], ignore_index=True)
        cols = list(collected[platform].columns) if platform in collected else []
        log(f"{platform}: {len(items)} rows scraped")
        return f"Scraped {len(items)} rows. Columns: {cols[:40]}"

    def execute(name, args):
        if name == "scrape_amazon_reviews":
            items = _run_actor(ACTORS["amazon"], {
                "productUrls": [{"url": u} for u in args["product_urls"]],
                "maxReviews": max_reviews, "sort": "recent"}, max_charge_usd)
            return store("Amazon", items)
        if name == "scrape_ebay_reviews":
            items = _run_actor(ACTORS["ebay"], {
                "product_urls": args["product_urls"], "reviews_limit": max_reviews}, max_charge_usd)
            return store("eBay", items)
        if name == "find_tiktok_videos":
            items = _run_actor(ACTORS["tiktok_search"], {
                "searchQueries": [args["query"]], "resultsPerPage": 15, "searchSection": "/video"},
                max_charge_usd)
            videos = [f"{i.get('webVideoUrl')} | comments={i.get('commentCount')} | {str(i.get('text', ''))[:120]}"
                      for i in items if i.get("webVideoUrl")]
            log(f"TikTok: found {len(videos)} videos for '{args['query']}'")
            return "\n".join(videos) or "No videos found."
        if name == "scrape_tiktok_comments":
            per_video = max(1, max_reviews // len(args["video_urls"]))
            items = _run_actor(ACTORS["tiktok_comments"], {
                "postURLs": args["video_urls"], "commentsPerPost": per_video}, max_charge_usd)
            return store("TikTok", items)
        raise ValueError(f"Unknown tool {name}")

    client = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))
    tools = _tool_defs(platforms)
    messages = [{"role": "user", "content": f"Product / links: {request}\nPlatforms: {', '.join(platforms)}"}]

    for _ in range(20):
        response = client.beta.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM_PROMPT, tools=tools, messages=messages,
            output_config={"effort": "medium"}, betas=[FALLBACK_BETA], fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("The AI model declined this request.")
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "pause_turn":
            continue
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            summary = "\n".join(b.text for b in response.content if b.type == "text")
            return collected, summary

        results = []
        for call in tool_calls:
            log(f"Agent → {call.name}({call.input})")
            try:
                results.append({"type": "tool_result", "tool_use_id": call.id,
                                "content": execute(call.name, call.input)})
            except Exception as e:
                log(f"{call.name} failed: {e}")
                results.append({"type": "tool_result", "tool_use_id": call.id,
                                "content": f"Error: {e}", "is_error": True})
        messages.append({"role": "user", "content": results})

    return collected, "Stopped after too many steps; returning what was collected."
