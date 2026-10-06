"""
Quick crawl4ai smoke-test: scrape 3 UTD pages and print a summary.
"""
import asyncio
import json
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

URLS = [
    ("utd_home",        "https://www.utdallas.edu/"),
    ("utd_academics",   "https://www.utdallas.edu/academics/"),
    ("utd_tuition",     "https://bursar.utdallas.edu/tuition/tuition-and-fees-schedule/"),
]

OUTPUT_DIR = Path("crawl4ai_output")
OUTPUT_DIR.mkdir(exist_ok=True)

async def scrape(crawler: AsyncWebCrawler, name: str, url: str) -> dict:
    print(f"\n→ Scraping [{name}] {url}")
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,
        remove_overlay_elements=True,
    )
    result = await crawler.arun(url=url, config=config)

    # Save raw markdown
    md_path = OUTPUT_DIR / f"{name}.md"
    md_path.write_text(result.markdown or "", encoding="utf-8")

    word_count = len((result.markdown or "").split())
    link_count = len(result.links.get("internal", [])) + len(result.links.get("external", []))

    summary = {
        "name": name,
        "url": url,
        "success": result.success,
        "status_code": result.status_code,
        "word_count": word_count,
        "internal_links": len(result.links.get("internal", [])),
        "external_links": len(result.links.get("external", [])),
        "media_images": len(result.media.get("images", [])),
        "markdown_preview": (result.markdown or "")[:500].replace("\n", " "),
        "output_file": str(md_path),
    }
    return summary

async def main():
    summaries = []
    async with AsyncWebCrawler() as crawler:
        for name, url in URLS:
            summary = await scrape(crawler, name, url)
            summaries.append(summary)
            print(f"  ✓ {summary['status_code']} | {summary['word_count']} words | "
                  f"{summary['internal_links']} internal links | "
                  f"{summary['media_images']} images")

    # Write JSON summary
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for s in summaries:
        print(f"\n[{s['name']}]")
        print(f"  URL:        {s['url']}")
        print(f"  Success:    {s['success']}  (HTTP {s['status_code']})")
        print(f"  Words:      {s['word_count']}")
        print(f"  Links:      {s['internal_links']} internal / {s['external_links']} external")
        print(f"  Images:     {s['media_images']}")
        print(f"  Output:     {s['output_file']}")
        print(f"  Preview:    {s['markdown_preview'][:200]}…")

    print(f"\nFull markdown saved to: {OUTPUT_DIR.resolve()}/")
    print(f"Summary JSON:          {summary_path.resolve()}")

asyncio.run(main())
