"""
Makes the week's second sheet: a copy of the For report in which every vehicle tab carries the trained-tagging
pipeline's values instead of the old pipeline's, so analysts can review and upload from it. The default sheet is
left as it is, as the fallback.

In the copy, for each vehicle tab:
  - country_code, ownership_status (the first column of that name) and validation_auto take the new pipeline's
    values; confidence and reasoning are cleared (they explain the old autovalidation);
  - tags: per post, a row keeps its place if the new pipeline gave the same tag; the post's other rows take its
    remaining tags; extra tags become copies of the post's last row (the upload creates a new tag for a repeated
    id-3); rows left over are deleted (the upload leaves those tags unkept);
  - posts the new pipeline did not tag keep their rows and tags; posts it did not put in this tab get
    validation_auto "Not in tab".

The results are read from S3 (exports/ford/trained_tagging/ and exports/ford/improved_upstream/), written by the
platform's ford-trained-tagging pipeline after the Monday export. Runs last in run_all.py, after autovalidation. If
this week's results are not there yet it waits (NEW_PIPELINE_WAIT_MINUTES, default 90), then gives up without making
a copy. The copy's link is saved in spreadsheet_links.csv (column new_model_url); if the week already has one, nothing
is done unless --force (which makes a fresh copy).

    python weekly_processing/new_model_sheet.py [--week YYYY-MM-DD] [--no-wait] [--force]
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import datetime
import io
import os
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

BUCKET = "fluency-data-platform"
TAGS_PREFIX = "exports/ford/trained_tagging/"
UPSTREAM_PREFIX = "exports/ford/improved_upstream/"
WAIT_MINUTES = int(os.getenv("NEW_PIPELINE_WAIT_MINUTES", "90"))
POLL_MINUTES = 5
MIN_MATCH_SHARE = 0.5  # below this the results are probably for a different week: make no copy
WRITE_CHUNK = 200      # rows per write request (rows carry long post text)

SET_COLS = ["country_code", "ownership_status", "validation_auto", "feedback_subcategory", "feedback_sentiment"]
CLEAR_COLS = ["confidence", "reasoning"]
NEEDED_COLS = SET_COLS + CLEAR_COLS + ["Validation", "id-3", "url", "brand", "model"]

# Older names of tags that may still appear in a sheet -> the names the pipeline uses
TAG_ALIASES = {
    "Overall Satisfaction With the Car": "Overall Satisfaction with the Vehicle",
    "Overall Satisfaction with the Car": "Overall Satisfaction with the Vehicle",
    "Range": "Range/Consumption",
    "Consumption/Range": "Range/Consumption",
    "Switching to Another Brand": "Switching to Another Vehicle",
    "Overall Disappointment With the Brand": "Overall Satisfaction with the Brand",
    "Communication From Brand": "Communication from Brand",
    "Aftersales (OEM App Support)": "OEM App Support Team",
    "Owners Manual": "Owner's Manual",
    "Communication With Dealer": "Communication with Dealer",
    "Infotainment": "Infotainment/Centerstack",
}


# ------------------------------------------------------------------
# Finding and reading the week's results
# ------------------------------------------------------------------

def _keys(s3, prefix: str, suffix: str) -> list[str]:
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(suffix)]
    return keys


def _stem(key: str) -> str:
    """'exports/ford/trained_tagging/2026-11-09_163643_verdicts.parquet' -> '2026-11-09_163643'"""
    return "_".join(key.rsplit("/", 1)[-1].split("_")[:2])


def find_results(s3, monday_str: str) -> dict | None:
    """S3 keys of the newest complete run made in the week starting monday_str, or None if there is none yet."""
    end = (datetime.date.fromisoformat(monday_str) + datetime.timedelta(days=7)).isoformat()
    in_week = lambda stem: monday_str <= stem[:10] < end
    verdicts = {_stem(k): k for k in _keys(s3, TAGS_PREFIX, "_verdicts.parquet") if in_week(_stem(k))}
    feedback = {_stem(k): k for k in _keys(s3, TAGS_PREFIX, "_feedback.parquet")}
    complete = sorted(st for st in verdicts if st in feedback)  # feedback is written last
    if not complete:
        return None
    run = complete[-1]
    mentions = sorted((_stem(k), k) for k in _keys(s3, UPSTREAM_PREFIX, "_vehicle_mentions.parquet")
                      if in_week(_stem(k)) and _stem(k) <= run)
    return {"run": run, "verdicts": verdicts[run], "feedback": feedback[run],
            "mentions": mentions[-1][1] if mentions else None}


def _read_parquet(s3, key: str, columns: list[str]) -> pd.DataFrame:
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body), columns=columns)


def read_results(s3, keys: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verdicts = _read_parquet(s3, keys["verdicts"], ["url", "tab", "validation_auto"])
    feedback = _read_parquet(s3, keys["feedback"],
                             ["url", "brand", "model", "ownership_status", "feedback_subcategory", "feedback_sentiment"])
    mentions = (_read_parquet(s3, keys["mentions"], ["url", "country_code"]) if keys["mentions"]
                else pd.DataFrame(columns=["url", "country_code"]))
    return verdicts, feedback, mentions


# ------------------------------------------------------------------
# Planning each tab's rows
# ------------------------------------------------------------------

def _clean(x) -> str:
    return str(x).strip() if pd.notna(x) and str(x).strip().lower() not in ("", "nan", "none") else ""


def _id(x) -> str:
    """id-3 as text, whether the sheet gives it as a number or a string."""
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return _clean(x)


def build_lookup(verdicts: pd.DataFrame, feedback: pd.DataFrame, mentions: pd.DataFrame) -> dict:
    """verdict per (tab, url); ownership and tag -> {(brand, model): sentiment} per url; country per url."""
    posts = {}
    for url, g in feedback.groupby("url", sort=False):
        own = next((o for o in map(_clean, g.ownership_status) if o), "")
        tags = {}  # a post naming two vehicles can give one tag two sentiments
        for b, m, t, se in zip(g.brand, g.model, g.feedback_subcategory, g.feedback_sentiment):
            if _clean(t):
                tags.setdefault(_clean(t), {})[(_clean(b), _clean(m))] = _clean(se) or "Neutral"
        posts[url] = {"own": own, "tags": tags}
    return {"verdict": {(t, u): v for t, u, v in zip(verdicts.tab, verdicts.url, verdicts.validation_auto)},
            "post": posts,
            "country": {u: _clean(c) for u, c in zip(mentions.url, mentions.country_code) if _clean(c)},
            "tabs": set(verdicts.tab)}


def _sentiment(by_vehicle: dict, brand: str, model: str) -> str:
    """The tag's sentiment for the row's own vehicle, else the first one given."""
    return by_vehicle.get((brand, model)) or next(iter(by_vehicle.values()))


def plan_tab(lookup: dict, tab: str, header: list[str], body: list[list]) -> tuple[list[list], dict]:
    """The tab's new rows (full rows, in order) and counts of what changed."""
    col = {name: header.index(name) for name in NEEDED_COLS}  # first column of each name
    urls = [_clean(r[col["url"]]) for r in body]
    rows_of = {}  # url -> its rows, in sheet order
    for i, u in enumerate(urls):
        rows_of.setdefault(u, []).append(i)

    new_tag, extra = {}, {}  # row -> our tag (None = drop the row); last row of a post -> tags to add after it
    stats = {"rows_in": len(body), "kept_tag": 0, "retagged": 0, "added": 0, "dropped": 0, "posts_no_tags": 0}
    for u, rows in rows_of.items():
        tags = lookup["post"].get(u, {}).get("tags") if u else None
        if not tags:
            stats["posts_no_tags"] += bool(u)
            continue
        left, free = list(tags), []
        for r in rows:
            cur = _clean(body[r][col["feedback_subcategory"]])
            cur = TAG_ALIASES.get(cur, cur)
            if cur in left:
                new_tag[r] = cur; left.remove(cur); stats["kept_tag"] += 1
            else:
                free.append(r)
        for r in free:
            if left:
                new_tag[r] = left.pop(0); stats["retagged"] += 1
            else:
                new_tag[r] = None; stats["dropped"] += 1
        if left:
            extra[rows[-1]] = left; stats["added"] += len(left)

    out = []
    for r, src in enumerate(body):
        u = urls[r]
        post = lookup["post"].get(u, {})
        base = list(src)
        base[col["country_code"]] = lookup["country"].get(u) or src[col["country_code"]]
        base[col["ownership_status"]] = post.get("own") or src[col["ownership_status"]]
        if u:
            base[col["validation_auto"]] = lookup["verdict"].get((tab, u), "Not in tab")
        for name in CLEAR_COLS:
            base[col[name]] = ""
        tags = ([] if new_tag.get(r, "") is None else [new_tag.get(r)]) + extra.get(r, [])
        brand, model = _clean(src[col["brand"]]), _clean(src[col["model"]])
        for tag in tags:
            row = list(base)
            if tag:  # our tag; a post we did not tag keeps the row's own tag and sentiment
                row[col["feedback_subcategory"]] = tag
                row[col["feedback_sentiment"]] = _sentiment(post["tags"][tag], brand, model)
            out.append(row)
    stats["rows_out"] = len(out)
    stats["not_in_tab"] = sum(row[col["validation_auto"]] == "Not in tab" for row in out)
    return out, stats


# ------------------------------------------------------------------
# Reading and writing the copy
# ------------------------------------------------------------------

def read_tab_raw(sheet, tab: str) -> tuple[list[str], list[list]]:
    """Header and rows as stored (numbers stay numbers, dates stay serial numbers), padded to the same width,
    so writing them back with RAW leaves every untouched cell exactly as it was."""
    result = sheet._sheets.spreadsheets().values().get(
        spreadsheetId=sheet.spreadsheet_id, range=f"'{tab}'",
        valueRenderOption="UNFORMATTED_VALUE", dateTimeRenderOption="SERIAL_NUMBER").execute()
    rows = result.get("values", [])
    if not rows:
        return [], []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return [str(h) for h in rows[0]], rows[1:]


def set_rows(sheet, tab: str, header: list[str], rows: list[list], n_now: int) -> None:
    """Make the tab's data rows exactly `rows`, given it has n_now data rows now. New rows are inserted above the
    last data row so they fall inside a Table if the tab is one, and get the Validation dropdown."""
    tab_id = sheet.tab_ids[tab]
    if len(rows) > n_now:
        sheet.batch_update([{"insertDimension": {
            "range": {"sheetId": tab_id, "dimension": "ROWS", "startIndex": n_now, "endIndex": len(rows)},
            "inheritFromBefore": True}}])
        v = header.index("Validation")
        sheet.batch_update([{"copyPaste": {
            "source": {"sheetId": tab_id, "startRowIndex": 1, "endRowIndex": 2, "startColumnIndex": v, "endColumnIndex": v + 1},
            "destination": {"sheetId": tab_id, "startRowIndex": 1, "endRowIndex": 1 + len(rows),
                            "startColumnIndex": v, "endColumnIndex": v + 1},
            "pasteType": "PASTE_DATA_VALIDATION"}}])
    sheet.write_values(tab, 1, 0, rows, chunk_size=WRITE_CHUNK)
    if len(rows) < n_now:
        sheet.batch_update([{"deleteDimension": {
            "range": {"sheetId": tab_id, "dimension": "ROWS", "startIndex": 1 + len(rows), "endIndex": 1 + n_now}}}])


def fill_tab(sheet, lookup: dict, tab: str) -> str:
    header, body = read_tab_raw(sheet, tab)
    missing = [c for c in NEEDED_COLS if c not in header]
    if not body or missing:
        return f"skipped ({'empty' if not body else 'no column ' + ', '.join(missing)}), old values left in"
    rows, st = plan_tab(lookup, tab, header, body)
    try:
        set_rows(sheet, tab, header, rows, len(body))
    except Exception as e:
        _, now = read_tab_raw(sheet, tab)  # the failure may have come before or after rows were added or removed
        set_rows(sheet, tab, header, body, len(now))  # put the tab back as it was
        return f"FAILED ({e}); tab restored with the old values"
    return (f"{st['rows_in']} rows -> {st['rows_out']} (tag kept {st['kept_tag']}, retagged {st['retagged']}, "
            f"added {st['added']}, dropped {st['dropped']}; untagged posts {st['posts_no_tags']}, "
            f"not in tab {st['not_in_tab']})")


def match_share(sheet, lookup: dict, tabs: list[str]) -> tuple[int, int]:
    """How many of the sheet's (tab, post) pairs the run put in that tab: a check that the run is this week's."""
    matched = total = 0
    for tab in tabs:
        header, body = read_tab_raw(sheet, tab)
        if "url" not in header:
            continue
        urls = {_clean(r[header.index("url")]) for r in body} - {""}
        total += len(urls)
        matched += sum((tab, u) in lookup["verdict"] for u in urls)
    return matched, total


# ------------------------------------------------------------------

def main(monday_str: str | None = None, wait: bool = True, force: bool = False) -> str | None:
    import boto3
    from auth import get_creds, get_service_account_email
    from google_sheet_processor import GoogleSheetProcessor, extract_file_id
    from helper_functions import get_monday_str, get_spreadsheet_url, get_new_model_url, save_new_model_url
    from ad_hoc.collect_report_sheets import add_new_sheet as add_to_report_folder

    monday_str = monday_str or get_monday_str()
    url = get_spreadsheet_url(monday_str)
    if url is None:
        print(f"No sheet for the week of {monday_str} in spreadsheet_links.csv — no new-model sheet made.")
        return None
    if get_new_model_url(monday_str) and not force:
        print(f"The week of {monday_str} already has a new-model sheet: {get_new_model_url(monday_str)} "
              "(--force makes a fresh copy).")
        return None

    s3 = boto3.client("s3")
    deadline = time.time() + (WAIT_MINUTES * 60 if wait else 0)
    while (keys := find_results(s3, monday_str)) is None and time.time() < deadline:
        print(f"  New pipeline results for {monday_str} not in S3 yet; checking again in {POLL_MINUTES} min.")
        time.sleep(POLL_MINUTES * 60)
    if keys is None:
        print(f"No new pipeline results for the week of {monday_str} — no new-model sheet made. "
              f"Rerun later with: python weekly_processing/new_model_sheet.py --week {monday_str}")
        return None
    print(f"Using new pipeline run {keys['run']}")
    lookup = build_lookup(*read_results(s3, keys))

    creds = get_creds()
    default = GoogleSheetProcessor(extract_file_id(url), creds)
    vehicle_tabs = pd.read_csv(PROJECT_ROOT / "vehicle_ids.csv")["vehicle_name"].tolist()
    tabs = [t for t in vehicle_tabs if t in default.tab_ids and t in lookup["tabs"]]
    matched, total = match_share(default, lookup, tabs)
    if total and matched / total < MIN_MATCH_SHARE:
        print(f"Only {matched:,} of {total:,} tab posts are in run {keys['run']} — probably a different week. "
              "No new-model sheet made.")
        return None

    copy = GoogleSheetProcessor.from_template(url, f"For report_{monday_str} (new model)", creds)
    copy.share_with(get_service_account_email())
    add_to_report_folder(copy.spreadsheet_id, creds)
    print(f"New-model sheet: {copy.url} ({matched:,} of {total:,} tab posts in the run)")

    failed = []
    for tab in tabs:
        result = fill_tab(copy, lookup, tab)
        print(f"  {tab}: {result}")
        if not result[0].isdigit():
            failed.append(tab)
    save_new_model_url(monday_str, copy.url)
    if failed:
        print(f"Done, but these tabs still hold the old pipeline's values: {', '.join(failed)}")
    else:
        print(f"Done: {len(tabs)} tabs filled.")
    return copy.url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--week", help="Monday of the sheet's week, YYYY-MM-DD (default: this week)")
    parser.add_argument("--no-wait", action="store_true", help="don't wait for the results if they are not in S3 yet")
    parser.add_argument("--force", action="store_true", help="make a fresh copy even if the week already has one")
    args = parser.parse_args()
    main(args.week, wait=not args.no_wait, force=args.force)
