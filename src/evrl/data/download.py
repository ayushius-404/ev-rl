"""Download ACN-Data sessions (Caltech) to data/raw/.

TODO(YOU) - this file needs you, because it needs YOUR account:
  1. Register for ACN-Data (search "ACN-Data Caltech") and request a free API token.
  2. Put the token in an environment variable:  export ACN_API_TOKEN=...   (Windows: set ACN_API_TOKEN=...)
  3. Read the CURRENT API docs and check that the URL, auth style and the `where` filter below are still
     correct. I wrote this from memory of the documentation, so treat it as a starting point, not gospel.
  4. If the API gives you trouble, the site also lets you download a JSON export manually. Just drop the file
     in data/raw/ and point `data.raw_path` in configs/default.yaml at it. The rest of the pipeline accepts
     both the API format (a dict with "_items") and a CSV with the same column names.
  5. Check the dataset's terms of use / citation requirement and add the citation to your report.

Usage (after the above):
    python -m evrl.data.download --site caltech --start 2019-01-01 --end 2019-12-31
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import requests

BASE = "https://ev.caltech.edu/api/v1/"  # TODO(YOU): verify against the current docs


def _http_date(d: str) -> str:
    return datetime.strptime(d, "%Y-%m-%d").strftime("%a, %d %b %Y 00:00:00 GMT")


def download(site: str, start: str, end: str, out_path: str, token: str) -> int:
    url = f"{BASE}sessions/{site}"
    params = {"where": f'connectionTime>="{_http_date(start)}" and connectionTime<="{_http_date(end)}"'}
    items: list[dict] = []
    while url:
        r = requests.get(url, params=params, auth=(token, ""), timeout=60)  # TODO(YOU): verify auth style
        r.raise_for_status()
        page = r.json()
        items.extend(page.get("_items", []))
        nxt = page.get("_links", {}).get("next", {}).get("href")
        url = None if not nxt else (nxt if nxt.startswith("http") else BASE + nxt.lstrip("/"))
        params = None  # the "next" link already carries the query
        print(f"  fetched {len(items)} sessions so far")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"_items": items}, f)
    return len(items)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="caltech")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", default="data/raw/sessions.json")
    a = ap.parse_args()
    tok = os.environ.get("ACN_API_TOKEN")
    if not tok:
        raise SystemExit("Set the ACN_API_TOKEN environment variable first (see the docstring of this file).")
    n = download(a.site, a.start, a.end, a.out, tok)
    print(f"Saved {n} sessions to {a.out}")
