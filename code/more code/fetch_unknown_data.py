"""
This scipt fetches unknown sequences for future prediction by our models
"""
import argparse
import os
import re
import sys
import time
from typing import Optional

import requests

START_URL = ("https://rest.uniprot.org/uniprotkb/search?"
             "format=fasta&"
             "query=%28length%3A%5B50+TO+150%5D+AND+%28protein_name%3A%22Uncharacterized+protein%22+OR+cc_function%3A%22unknown+function%22%29+AND+fragment%3Afalse+AND+keyword%3A%22Reference+proteome%22%29&"
             "size=500")

HEADERS = {"Accept": "text/fasta", "User-Agent": "uniprot-fetch/1.0"}

def parse_next(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    for part in link_header.split(","):
        m = re.search(r'<([^>]+)>\s*;\s*rel="?next"?', part)
        if m:
            return m.group(1)
    return None

def fetch_all_fasta(out_path: str, timeout: int = 60, max_retries: int = 5):
    url = START_URL
    first = True
    total_bytes = 0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as fout:
        while url:
            tries = 0
            while True:
                try:
                    resp = requests.get(url, headers=HEADERS, timeout=timeout)
                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get("Retry-After", "3"))
                        time.sleep(retry_after)
                        continue
                    resp.raise_for_status()
                    break
                except Exception as e:
                    tries += 1
                    if tries >= max_retries:
                        raise
                    time.sleep(min(2 ** tries, 60))
            data = resp.content
            if first and data.startswith(b"\xef\xbb\xbf"):
                data = data[3:]
            fout.write(data)
            total_bytes += len(data)
            next_url = parse_next(resp.headers.get("Link"))
            url = next_url
            first = False
    return total_bytes

def main():
    ap = argparse.ArgumentParser(description="Fetch full UniProtKB FASTA for a query into a single file (cursor-paginated).")
    ap.add_argument("--out", default="uniprot_unknown_50_150_refprot.fasta", help="Output FASTA file path")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout (s)")
    args = ap.parse_args()
    try:
        nbytes = fetch_all_fasta(args.out, timeout=args.timeout)
        print(f"Done. Wrote {nbytes/1024:.1f} KB to {os.path.abspath(args.out)}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
