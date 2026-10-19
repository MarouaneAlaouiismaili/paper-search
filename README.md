# Paper Search

**A transparent command-line research tool for academic literature and the public web.**

Search across configurable sources, refine results with Boolean queries and metadata filters, and save evidence reports with source links and field-level provenance. Paper Search uses public HTTP(S) pages and official APIs; it does not bypass sign-in, CAPTCHA, or access controls.

> **Python 3.9+** · Install with pip · Use as a CLI or Python module

## Quick start

~~~sh
python -m pip install .
mkdir -p reports
paper-search '"climate researcher"' --mode investigation --report reports/case.md --llm
~~~

Install dependencies directly with `python -m pip install -r requirements.txt`. You can also run the module with `python -m paper_search.cli ...`.

## What it does

- Searches academic indexes and public-web sources in one workflow.
- Supports Boolean expressions, fielded queries, source selection, and metadata filters.
- Combines local BM25 ranking with Reciprocal Rank Fusion across sources.
- Optionally expands OpenAlex results through a one-step citation network search.
- Produces Markdown or JSON evidence reports and CSV or JSON exports.
- Can fetch selected public pages and optionally summarize evidence with a local OpenAI-compatible model.
- Keeps provenance, missing metadata, source errors, and discovery paths visible for review.

## Search modes and examples

Available modes are `investigation`, `academic`, and `web`. The default mode searches available academic and public-web sources, requesting 20 results per page across 3 pages per source. Set `--num` and `--pages` to change this; the limits are 100 results per page and 10 pages per source.

~~~sh
# Search academic literature
paper-search 'climate change AND policy' --mode academic

# Search the public web for a person on a specific site
paper-search '"climate researcher"' --mode web --type person --site example.org

# Select academic sources explicitly
paper-search 'LLM OR "large language model"' --source openalex --source arxiv
~~~

## Search providers

### Google Custom Search

The Google connector uses the official Custom Search JSON API; it does not scrape Google result pages. Configure an existing Programmable Search Engine with `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`. When both are set, Google is included automatically in `web` and `investigation` modes; use `--source google` to request it explicitly. Results depend on the search engine configuration.

~~~sh
export GOOGLE_API_KEY='your-api-key'
export GOOGLE_CSE_ID='your-search-engine-id'
paper-search 'climate change policy' --mode web --source google
~~~

Google is no longer accepting new customers for this API. Existing customers can use it through January 1, 2027, and charges may apply beyond the free quota. Without these credentials, web search can continue with DuckDuckGo.

### Brave Search

Set `BRAVE_SEARCH_API_KEY` to enable the official Brave Web Search API. Brave is then included automatically in `web` and `investigation` modes. The connector supports up to 20 results per API request and 10 API pages. Each query, page, or site restriction can create a separate request; check [Brave's current pricing](https://brave.com/search/api/) before running large searches.

~~~sh
export BRAVE_SEARCH_API_KEY='your-brave-search-api-key'
paper-search 'open source researcher' --mode web --site example.org --pages 3
~~~

Store API keys in environment variables, not in commands or the repository. Search queries are sent to the selected provider. See the [Brave privacy policy](https://api-dashboard.search.brave.com/privacy-policy) for query logging and retention terms. DuckDuckGo can return CAPTCHA pages; Paper Search does not attempt to bypass them.

### Reverse image search

Use Google Cloud Vision Web Detection to find exact and partial matches, pages containing matching images, and visually similar images.

~~~sh
mkdir -p reports exports
export GOOGLE_CLOUD_VISION_API_KEY='your-restricted-api-key'
paper-search --image ./photo.jpg --image-max-results 10 \
  --report reports/image-report.md --export exports/image-matches.csv
~~~

Enable the Cloud Vision API and billing, and restrict the key to that API. The image file is sent to Google Cloud for analysis; do not use private images unless you are authorized to share them. Supported formats are JPEG, PNG, GIF, BMP, WebP, TIFF, and ICO, up to 7 MiB. See Google's [Web Detection guide](https://docs.cloud.google.com/vision/docs/detecting-web), [supported files](https://docs.cloud.google.com/vision/docs/supported-files), and [pricing](https://cloud.google.com/vision/pricing).

Reports distinguish exact, partial, and visual matches. Visual similarity does not establish that two images are the same, and no result only means the service found no reference in its index.

## Query language and filters

Use `AND`, `OR`, `NOT`, parentheses, and quoted phrases. Precedence is `NOT`, then `AND`, then `OR`; adjacent terms mean `AND`.

~~~text
(LLM OR "large language model") AND NOT blockchain
title:"climate change" AND year:2020..2025
site:example.edu AND type:person
citations:>=100 AND open_access:true
~~~

Supported CLI filters include source, record type, site, year range, author, language, open-access status, and minimum citations. Record types are `paper`, `person`, `organization`, `website`, `news`, `social`, and `other`. Web-page classification is approximate; review the source link and snippet before drawing conclusions.

Each `OR` branch must contain a positive search term. Queries expand to at most 8 search forms per source; broader forms return a clear error rather than silently dropping terms. Simple multi-word searches may also include partial matches, which are labeled for review. Use `--strict` to disable partial matches. By default, results missing year, author, type, language, citation, or open-access metadata are kept and labeled; use `--unknown-metadata exclude` to omit them. Site and source filters always require an exact match.

## Fetching, ranking, and reports

- **Ranking:** The combined search uses local BM25 and Reciprocal Rank Fusion. Citation count and publication date do not add relevance points.
- **Citation expansion:** In `investigation` and `academic` modes, the default OpenAlex expansion follows references and citing works for up to 12 records. Set `--citation-expansion-limit 0..20`; use 0 to turn it off.
- **Deep fetch:** `--deep-pages` reads up to 25 selected public pages (15 by default), one page deep, with a 1 MiB and 8-second limit per page. Redirects are not followed. The tool extracts main content where possible; full page contents are not sent to the optional model.
- **Evidence report:** `--report reports/case.md` or `--report reports/case.json` records filters, links, snippets, field provenance, citation expansion paths, missing or conflicting values, and source errors.
- **Data export:** `--export exports/results.csv` or `--export exports/results.json` saves result data. Create the output directories first. Generated reports and exports may contain sensitive search data and are ignored by Git.

A failed request from every source is reported as an incomplete search, not as a confirmed absence of results.

## Optional local LLM summary

With `--llm`, the best five useful records are sent to a local OpenAI-compatible server. Evidence is limited to 7,500 characters and the default response to 400 tokens. The model summarizes snippets; it does not independently verify identities or claims.

~~~sh
export LLM_API_KEY='your-local-api-key'
paper-search '"climate change policy"' --mode investigation --llm
~~~

The default model is `unsloth/Qwen3-4B-GGUF:UD-Q4_K_XL` at `http://127.0.0.1:8888/v1/chat/completions`. Configure `LLM_MODEL` and `LLM_API_URL` or pass `--llm-model` and `--llm-url`. Internal reasoning is off by default; `--llm-thinking` enables it.

## Responsible use

Paper Search is designed for research and review. It accesses public HTTP(S) pages and official APIs only, respects access controls, and does not bypass CAPTCHA or sign-in. Results can be incomplete, stale, or incorrectly classified; check the cited sources before relying on them. Follow each provider's terms and use only data and images you are authorized to search.
