from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen
import argparse
import json
import re
import shutil
import time

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "sign_letters"

ENGLISH_CATEGORY = "ASL_letters"
ARABIC_CATEGORY = "Arab_manual_alphabet"


@dataclass(frozen=True)
class ScrapedAsset:
    language: str
    letter: str | None
    file_title: str
    page_url: str
    image_url: str
    output_path: str
    license_name: str | None = None
    source: str | None = None


def _api_get(params: dict[str, Any]) -> dict[str, Any]:
    query = urlencode({**params, "format": "json", "formatversion": 2})
    request = Request(f"{WIKIMEDIA_API}?{query}", headers={"User-Agent": "arsl-person1-scraper/1.0"})
    delay_seconds = 1.0
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            last_error = error
            if error.code != 429:
                raise
        except URLError as error:
            last_error = error
        time.sleep(delay_seconds)
        delay_seconds *= 2
    if last_error is not None:
        raise last_error


def _iter_category_files(category_title: str) -> list[str]:
    members: list[str] = []
    continuation: dict[str, Any] = {}
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category_title}",
            "cmtype": "file",
            "cmlimit": 50,
        }
        params.update(continuation)
        data = _api_get(params)
        members.extend(item["title"] for item in data.get("query", {}).get("categorymembers", []))
        continuation = data.get("continue", {})
        if not continuation:
            break
    return members


def _extmeta_value(extmetadata: dict[str, Any], key: str) -> str | None:
    value = extmetadata.get(key)
    if not isinstance(value, dict):
        return None
    text = value.get("value")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def _fetch_file_infos(file_titles: list[str]) -> dict[str, dict[str, Any]]:
    data = _api_get(
        {
            "action": "query",
            "prop": "imageinfo",
            "titles": "|".join(file_titles),
            "iiprop": "url|extmetadata",
        }
    )
    pages = data.get("query", {}).get("pages", [])
    infos: dict[str, dict[str, Any]] = {}
    for page in pages:
        if "imageinfo" not in page:
            continue
        info = page["imageinfo"][0]
        extmetadata = info.get("extmetadata", {})
        page_title = page.get("title")
        if not isinstance(page_title, str):
            continue
        infos[page_title] = {
            "page_title": page_title,
            "page_url": f"https://commons.wikimedia.org/wiki/{quote(page_title.replace(' ', '_'))}",
            "image_url": info["url"],
            "license_name": _extmeta_value(extmetadata, "LicenseShortName"),
            "source": _extmeta_value(extmetadata, "ImageDescription") or _extmeta_value(extmetadata, "ObjectName"),
        }
    if not infos:
        raise RuntimeError(f"No metadata returned for files: {', '.join(file_titles)}")
    return infos


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "arsl-person1-scraper/1.0"})
    delay_seconds = 1.0
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urlopen(request, timeout=60) as response, destination.open("wb") as output:
                shutil.copyfileobj(response, output)
            return
        except HTTPError as error:
            last_error = error
            if error.code != 429:
                raise
        except URLError as error:
            last_error = error
        time.sleep(delay_seconds)
        delay_seconds *= 2
    if last_error is not None:
        raise last_error


def _safe_stem(title: str) -> str:
    stem = title.replace("File:", "")
    stem = stem.replace("/", "_")
    stem = re.sub(r"[\\:*?\"<>|]", "_", stem)
    stem = re.sub(r"\s+", "_", stem).strip("._")
    return stem or "asset"


def _infer_english_letter(file_title: str) -> str | None:
    match = re.match(r"^(?:File:)?Sign language ([A-Z])\.", file_title)
    if match:
        return match.group(1)
    match = re.match(r"^(?:File:)?ASL[_ ]([A-Z])(?:[@_\s]|$)", file_title)
    if match:
        return match.group(1)
    return None


def _select_english_letter_files(file_titles: list[str]) -> list[str]:
    by_letter: dict[str, list[str]] = {chr(code): [] for code in range(ord("A"), ord("Z") + 1)}
    for file_title in file_titles:
        letter = _infer_english_letter(file_title)
        if letter:
            by_letter.setdefault(letter, []).append(file_title)

    selected: list[str] = []
    for letter in map(chr, range(ord("A"), ord("Z") + 1)):
        candidates = by_letter.get(letter, [])
        if not candidates:
            continue
        chosen = sorted(candidates, key=lambda title: (len(title), title))[0]
        selected.append(chosen)
    return selected


def scrape_category(category_title: str, language: str, output_root: Path, per_letter: bool = False) -> list[ScrapedAsset]:
    file_titles = _iter_category_files(category_title)
    if per_letter:
        file_titles = _select_english_letter_files(file_titles)

    assets: list[ScrapedAsset] = []
    language_dir = output_root / language
    language_dir.mkdir(parents=True, exist_ok=True)

    info_by_title: dict[str, dict[str, Any]] = {}
    for start in range(0, len(file_titles), 20):
        batch = file_titles[start : start + 20]
        info_by_title.update(_fetch_file_infos(batch))

    for file_title in file_titles:
        info = info_by_title[file_title]
        letter = _infer_english_letter(file_title) if language == "english" else None
        filename = _safe_stem(file_title)
        suffix = Path(urlsplit(info["image_url"]).path).suffix or ".jpg"
        destination = language_dir / f"{filename}{suffix}"
        _download_file(info["image_url"], destination)
        time.sleep(1.0)
        assets.append(
            ScrapedAsset(
                language=language,
                letter=letter,
                file_title=info["page_title"],
                page_url=info["page_url"],
                image_url=info["image_url"],
                output_path=str(destination),
                license_name=info.get("license_name"),
                source=info.get("source"),
            )
        )
    return assets


def write_manifest(assets: list[ScrapedAsset], output_root: Path) -> Path:
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([asdict(asset) for asset in assets], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def scrape_both_languages(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_root = Path(output_dir)
    english_assets = scrape_category(ENGLISH_CATEGORY, "english", output_root, per_letter=True)
    arabic_assets = scrape_category(ARABIC_CATEGORY, "arabic", output_root, per_letter=False)
    return write_manifest([*english_assets, *arabic_assets], output_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape sign language letter assets from Wikimedia Commons.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where to store downloaded assets")
    parser.add_argument("--english-category", default=ENGLISH_CATEGORY)
    parser.add_argument("--arabic-category", default=ARABIC_CATEGORY)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_dir)
    english_assets = scrape_category(args.english_category, "english", output_root, per_letter=True)
    arabic_assets = scrape_category(args.arabic_category, "arabic", output_root, per_letter=False)
    manifest = write_manifest([*english_assets, *arabic_assets], output_root)
    print(f"Downloaded {len(english_assets)} English assets and {len(arabic_assets)} Arabic assets")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())