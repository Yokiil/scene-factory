"""Fetch source metadata for the JEV compatibility audit, without credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from html.parser import HTMLParser


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, text):
        if not self.skip:
            self.parts.append(text.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    request = urllib.request.Request(args.url, headers={"User-Agent": "SceneFactory-source-audit/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    source = data.decode("utf-8", errors="replace")
    if args.url.endswith(".md"):
        text, links = source, re.findall(r"https://[^\s)]+", source)
    else:
        parsed = Text()
        parsed.feed(source)
        text, links = " ".join(parsed.parts), parsed.links
    print(json.dumps({"url": args.url, "sha256": hashlib.sha256(data).hexdigest(),
                      "size_bytes": len(data), "links": links[:15]}, ensure_ascii=True))
    if not args.metadata_only:
        print(text[:50000].encode("ascii", "backslashreplace").decode())


if __name__ == "__main__":
    main()
