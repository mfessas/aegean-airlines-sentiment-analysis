"""Collect AirlineRatings reviews for the AEGEAN benchmarking project.

Run directly or from Airline Ratings scraper.ipynb. Install pandas,
beautifulsoup4, lxml and playwright, then run `python -m playwright install chromium`.
Raw reviews and resumable checkpoints are stored under Data/Scraping.
Analytical date, cabin and flight-duration filters belong to downstream processing.
"""
import json
import random
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ---------------- USER SETTINGS ----------------
SUPPORTED_AIRLINE_URLS = [
    'https://www.airlineratings.com/airlines/aegean-airlines/reviews?page=1&limit=50',
    'https://www.airlineratings.com/airlines/turkish-airlines/reviews?page=1&limit=50',
    'https://www.airlineratings.com/airlines/air-france/reviews?page=1&limit=50',
    'https://www.airlineratings.com/airlines/lufthansa/reviews?page=1&limit=50',
    'https://www.airlineratings.com/airlines/bangkok-air/reviews?page=1&limit=50',
    'https://www.airlineratings.com/airlines/porter-airlines/reviews?page=1&limit=50',
    'https://www.airlineratings.com/airlines/virgin-australia/reviews?page=1&limit=50',
]
# Select the airlines to scrape in this session; the combined dataset always
# includes saved checkpoints for all supported airlines.
AIRLINE_URLS = SUPPORTED_AIRLINE_URLS.copy()
# Example: AIRLINE_URLS = [SUPPORTED_AIRLINE_URLS[0]]  # AEGEAN only

MAX_PAGES = None            # None = all pages reported by site; integer = test page limit
MAX_REVIEWS = None          # e.g. 10 for testing; None for all discovered
REFRESH_EXISTING = False   # True to re-scrape already saved reviews
HEADLESS = False            # Visible browser, as in the working solution
DELAY_MIN = 4.1
DELAY_MAX = 9.9
# Resolve from the script location, not the caller's working directory.
PROJECT_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents
     if all((parent / name).is_dir() for name in ('Python Code', 'Data', 'Streamlit'))),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError('Place this script inside the AEGEAN project folder tree.')
OUTPUT_DIR = PROJECT_ROOT / 'Data' / 'Scraping' / 'airlineratings_data' 
COMBINED_CSV = OUTPUT_DIR / 'all_airlines_reviews.csv'
COMBINED_JSON = OUTPUT_DIR / 'all_airlines_reviews.json'
USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
              'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36')
# -----------------------------------------------
BASE = 'https://www.airlineratings.com'
LABELS = ['Overall Value for Money', 'Seat and Cabin Space', 'Customer Service',
          'In Flight Entertainment', 'Baggage Handling', 'Check-in Process', 'Meals and Beverages']


def normalize(s):
    """Collapse whitespace in extracted HTML text."""
    return re.sub(r'\s+', ' ', s or '').strip()


def airline_slug(url):
    """Extract the airline identifier from its review-listing URL."""
    match = re.search(r'/airlines/([^/?#]+)/reviews', url)
    if not match:
        raise ValueError('AIRLINE_URL must contain /airlines/<slug>/reviews')
    return match.group(1)


def page_url(url, number):
    """Construct a paginated listing URL while preserving other query parameters."""
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    q['page'] = [str(number)]
    q['limit'] = ['50']  # Confirmed working; limit=500 breaks pagination
    return urlunparse(parsed._replace(query=urlencode(q, doseq=True)))


class RateLimiter:
    """Space browser navigations using a shared randomized delay."""
    def __init__(self):
        """Initialize the timestamp of the previous navigation."""
        self.last = None

    def wait(self):
        """Wait for the remaining interval before the next navigation."""
        if self.last is not None:
            wait_for = random.uniform(DELAY_MIN, DELAY_MAX) - (time.monotonic() - self.last)
            if wait_for > 0:
                time.sleep(wait_for)
        self.last = time.monotonic()


def navigate(page, url, limiter):
    """Load a page and reject HTTP errors or unexpected redirects."""
    limiter.wait()
    response = page.goto(url, wait_until='domcontentloaded', timeout=60000)
    if response is None or response.status in (403, 429):
        raise RuntimeError(f'Access restricted: {url} ({response.status if response else "no response"})')
    if response.status >= 400:
        raise RuntimeError(f'HTTP {response.status}: {url}')
    if urlparse(page.url).path != urlparse(url).path:
        raise RuntimeError(f'Unexpected redirect to {page.url}')
    return response


def listing_rows(page, slug, requested_page, previous_page_ids=None):
    # The site sometimes initially renders a different airline and then hydrates.
    # Wait for the correct page AND a changed table, not just a changed page indicator.
    """Wait for the requested airline and page to hydrate, then extract unique review links and page count."""
    expected = slug.replace('-', ' ').casefold()
    previous_ids = sorted(str(x) for x in (previous_page_ids or []))
    try:
        page.wait_for_function(r'''({expected, requested, previousIds}) => {
          const nav = document.querySelector('nav[aria-label="pagination navigation"]');
          if (!nav || Number(nav.getAttribute('data-active-page')) !== requested) return false;
          const rows = [...document.querySelectorAll('tbody tr[role="row"]')];
          const matching = rows.filter(row => {
            const td = row.querySelector('td[role="rowheader"]');
            return td && td.innerText.toLowerCase().includes(expected);
          });
          if (!matching.length) return false;
          const ids = [...new Set(matching.flatMap(row =>
            [...row.querySelectorAll('a[href^="/reviews/"]')]
              .map(a => (a.getAttribute('href') || '').match(/^\/reviews\/(\d+)\/?$/)?.[1])
              .filter(Boolean)
          ))].sort();
          if (!ids.length) return false;
          return !(ids.length === previousIds.length &&
                   ids.every((id, i) => id === previousIds[i]));
        }''', arg={'expected': expected, 'requested': requested_page,
                   'previousIds': previous_ids}, timeout=45000)
    except PlaywrightTimeout:
        nav = page.locator('nav[aria-label="pagination navigation"]')
        state = nav.first.get_attribute('data-active-page') if nav.count() else 'missing'
        raise RuntimeError(
            f'Page {requested_page} did not show new review IDs within 45 seconds '
            f'(active={state}). The table may still be stale or the server returned duplicates.'
        )
    # Allow the hydrated table to settle before capturing the HTML.
    page.wait_for_timeout(900)

    soup = BeautifulSoup(page.content(), 'lxml')
    nav = soup.select_one('nav[aria-label="pagination navigation"]')
    active = int(nav.get('data-active-page')) if nav and (nav.get('data-active-page') or '').isdigit() else None
    total = int(nav.get('data-total')) if nav and (nav.get('data-total') or '').isdigit() else None
    if active != requested_page:
        raise RuntimeError(f'Wrong page: requested={requested_page}, active={active}')
    records = []
    seen_on_page = set()
    for tr in soup.select('tbody tr[role="row"]'):
        td = tr.select_one('td[role="rowheader"]')
        if not td or expected not in normalize(td.get_text(' ', strip=True)).casefold():
            continue
        link = tr.select_one('a[href^="/reviews/"]')
        if not link:
            continue
        href = link.get('href', '')
        if not re.fullmatch(r'/reviews/\d+/?', href):
            continue
        review_id = href.rstrip('/').split('/')[-1]
        if review_id in seen_on_page:
            continue
        seen_on_page.add(review_id)
        cells = tr.select('td')
        records.append({
            'review_id': review_id,
            'review_url': urljoin(BASE, href),
            'airline_listing': normalize(td.get_text(' ', strip=True)),
            'cabin_listing': normalize(cells[1].get_text(' ', strip=True)) if len(cells) > 1 else None,
            'excerpt_listing': normalize(cells[2].get_text(' ', strip=True)) if len(cells) > 2 else None,
            'rating_listing': normalize(cells[3].get_text(' ', strip=True)) if len(cells) > 3 else None,
        })
    return records, total


def extract_review(page, link_record, slug):
    """Extract review metadata, text and category ratings from the detail page. Preserve raw text for auditing."""
    soup = BeautifulSoup(page.content(), 'lxml')
    main = soup.select_one('main') or soup.body or soup
    # Heading is expected to contain the airline and route; avoids unrelated recommendations.
    heading = main.find('h1')
    title = normalize(heading.get_text(' ', strip=True)) if heading else ''
    expected = slug.replace('-', ' ').casefold()
    # Prefer heading check; if missing, look for a link back to the selected airline.
    own_airline_link = soup.select_one(f'a[href^="/airlines/{slug}"]')
    if expected not in title.casefold() and not own_airline_link:
        raise ValueError(f'Airline verification failed for {page.url}: {title[:100]}')

    # Main content only, and strip scripts, nav, footer, comments before text extraction.
    clone = BeautifulSoup(str(main), 'lxml')
    for element in clone.select('script, style, nav, footer'):
        element.decompose()
    lines = [normalize(x) for x in clone.get_text('\n', strip=True).splitlines()]
    lines = [x for x in lines if x]
    joined = '\n'.join(lines)

    record = dict(link_record)
    record.update({
        'airline': slug.replace('-', ' ').title(),
        'review_title': title or None,
        'review_date': None,
        'trip_date': None,
        'origin': None,
        'destination': None,
        'travel_class': None,
        'overall_rating': None,
        'reviewer_name': None,
        'review_text': None,
        'category_ratings': {},
        'raw_main_text': joined,
        'scraped_at_utc': datetime.now(timezone.utc).isoformat(),
    })

    m = re.search(r'\b(\d{1,2})\s*/\s*10\b', joined)
    if m:
        record['overall_rating'] = int(m.group(1))
    m = re.search(r'\b(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})\s+by\s+([^\n]+)', joined)
    if m:
        record['review_date'], record['reviewer_name'] = m.group(1), m.group(2)
    m = re.search(r'\b(?:Departure|Travel date|Trip date)\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})', joined, re.I)
    if m:
        record['trip_date'] = m.group(1)
    m = re.search(r'\bFrom\s+(.+?)\s+to\s+([^\n]+)', joined, re.I)
    if m:
        record['origin'], record['destination'] = m.group(1), m.group(2)
    m = re.search(r'\b(Economy class|Premium economy|Business class|First class)\b', joined, re.I)
    if m:
        record['travel_class'] = m.group(1)

    # Capture the review body between the route and subsequent rating/comments sections.
    # This is heuristic and raw_main_text is always preserved for audit.
    for idx, line in enumerate(lines):
        if re.match(r'^From\s+.+\s+to\s+.+$', line, re.I):
            body = []
            for next_line in lines[idx + 1:]:
                if next_line in ('Comments', 'Add a comment', 'View airline profile', 'See more reviews') or next_line in LABELS:
                    break
                if re.fullmatch(r'\d+', next_line):
                    break
                body.append(next_line)
            if body:
                record['review_text'] = normalize(' '.join(body))
            break

    # Sub-ratings are represented as a label <p> beside a progressbar.
    # Read aria-valuetext (actual score, e.g. '2'), NOT aria-valuenow
    # (percentage, e.g. '40'). Preserve every category dynamically.
    for progress in main.select('[role="progressbar"]'):
        container = progress.find_parent('div', class_=lambda c: c and 'items-center' in c.split() and 'border' in c.split())
        if container is None:
            # Fallback: nearest ancestor containing a sibling <p> label.
            for parent in progress.parents:
                if parent is main or parent.name in ('main', 'body', 'html'):
                    break
                label = parent.find('p', recursive=False)
                if label:
                    container = parent
                    break
        if container is None:
            continue
        label = container.find('p', recursive=False)
        if label is None:
            continue
        category = normalize(label.get_text(' ', strip=True))
        if not category:
            continue
        score = normalize(progress.get('aria-valuetext'))
        if not score:
            span = progress.select_one('span')
            score = normalize(span.get_text(' ', strip=True)) if span else ''
        if score.lower() in ('n/a', 'na', '-', ''):
            score = None
        else:
            try:
                score = float(score)
                if score.is_integer():
                    score = int(score)
            except ValueError:
                pass
        record['category_ratings'][category] = score

    # Fallback for other markup variants, without overwriting DOM ratings.
    for i, line in enumerate(lines[:-1]):
        if line in LABELS and line not in record['category_ratings']:
            value = lines[i + 1]
            record['category_ratings'][line] = None if value.lower() in ('n/a', 'na', '-') else value

    # Additional structured data, when supplied, can fill missing values.
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            payload = json.loads(script.string or script.get_text())
        except (ValueError, TypeError):
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict) or item.get('@type') != 'Review':
                continue
            record['review_text'] = record['review_text'] or item.get('reviewBody')
            record['review_date'] = record['review_date'] or item.get('datePublished')
            author = item.get('author')
            if isinstance(author, dict):
                record['reviewer_name'] = record['reviewer_name'] or author.get('name')
            rating = item.get('reviewRating')
            if isinstance(rating, dict) and record['overall_rating'] is None:
                record['overall_rating'] = rating.get('ratingValue')
    return record


def write_json_atomic(path, records):
    """Write JSON to a temporary sibling, then replace the completed file."""
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def write_csv_atomic(path, dataframe):
    """Replace a CSV only after the complete temporary file has been written."""
    temporary = path.with_name(path.name + '.tmp')
    dataframe.to_csv(temporary, index=False, encoding='utf-8-sig')
    temporary.replace(path)


def save(output, slug, links, reviews):
    """Persist per-airline links, review checkpoints and flattened CSV ratings."""
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / f'{slug}_links.json', links)
    write_json_atomic(output / f'{slug}_reviews.json', reviews)
    df = pd.DataFrame(reviews)
    if not df.empty:
        categories = sorted({k for r in reviews for k in (r.get('category_ratings') or {})})
        for category in categories:
            df['rating_' + re.sub(r'[^a-z0-9]+', '_', category.lower()).strip('_')] = [
                (r.get('category_ratings') or {}).get(category) for r in reviews
            ]
        df['category_ratings'] = df['category_ratings'].map(lambda x: json.dumps(x or {}, ensure_ascii=False))
    write_csv_atomic(output / f'{slug}_reviews.csv', df)


def scrape_airline(context, limiter, airline_url):
    """Discover review links, resume saved work and checkpoint each successful extraction."""
    slug = airline_slug(airline_url)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    links_path = OUTPUT_DIR / f'{slug}_links.json'
    reviews_path = OUTPUT_DIR / f'{slug}_reviews.json'
    links = json.loads(links_path.read_text(encoding='utf-8')) if links_path.exists() else []
    reviews = json.loads(reviews_path.read_text(encoding='utf-8')) if reviews_path.exists() else []
    seen = {str(x['review_id']) for x in links}
    completed = {str(x['review_id']) for x in reviews}
    page = context.new_page()
    try:
        # Revisit listing pages even when their links are already checkpointed.
        # A cached page with zero new links is NOT the end of pagination.
        first_page = int(parse_qs(urlparse(airline_url).query).get('page', ['1'])[0])
        page_number = first_page
        total_pages = None
        previous_page_ids = None
        while True:
            if MAX_PAGES is not None and page_number >= first_page + MAX_PAGES:
                break
            if total_pages is not None and page_number > total_pages:
                break
            url = page_url(airline_url, page_number)
            print(f'Listing page {page_number}: {url}', flush=True)
            navigate(page, url, limiter)
            # Retry the listing when React hydration or cached content is slow.
            for attempt in range(1, 4):
                try:
                    found, reported_total = listing_rows(
                        page, slug, page_number, previous_page_ids
                    )
                    break
                except RuntimeError as exc:
                    if attempt == 3:
                        raise
                    print(f'  Waiting/reloading page {page_number} '
                          f'(attempt {attempt}/3): {exc}', flush=True)
                    navigate(page, url, limiter)
            if reported_total is not None:
                total_pages = reported_total
            current_ids = {item['review_id'] for item in found}
            if not found:
                print('  No review rows; stopping discovery.', flush=True)
                break
            if previous_page_ids == current_ids:
                raise RuntimeError(f'Page {page_number} still duplicates previous page after waiting for table update')
            previous_page_ids = current_ids
            new = [item for item in found if item['review_id'] not in seen]
            for item in new:
                links.append(item)
                seen.add(item['review_id'])
            save(OUTPUT_DIR, slug, links, reviews)
            print(f'  Found {len(found)}, new {len(new)}, total links {len(links)}, '
                  f'page {page_number}/{total_pages or "?"}', flush=True)
            if MAX_REVIEWS is not None and len(links) >= MAX_REVIEWS:
                break
            page_number += 1

        queue = links[:MAX_REVIEWS] if MAX_REVIEWS else links
        for index, item in enumerate(queue, 1):
            if str(item['review_id']) in completed and not REFRESH_EXISTING:
                continue
            print(f'Review {index}/{len(queue)}: {item["review_url"]}', flush=True)
            try:
                navigate(page, item['review_url'], limiter)
                review = extract_review(page, item, slug)
            except Exception as exc:
                print(f'  Error: {exc}', flush=True)
                if 'Access restricted' in str(exc):
                    break
                continue
            reviews = [r for r in reviews if str(r['review_id']) != str(item['review_id'])]
            reviews.append(review)
            completed.add(str(item['review_id']))
            save(OUTPUT_DIR, slug, links, reviews)
    finally:
        page.close()
        save(OUTPUT_DIR, slug, links, reviews)
    print(f'Finished {slug}: {len(reviews)} reviews', flush=True)



def save_combined():
    """Rebuild combined outputs from per-airline checkpoints, including previous runs."""
    combined = {}
    for url in SUPPORTED_AIRLINE_URLS:
        slug = airline_slug(url)
        path = OUTPUT_DIR / f'{slug}_reviews.json'
        if not path.exists():
            continue
        try:
            records = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            print(f'Cannot load {path}: {exc}', flush=True)
            continue
        for record in records:
            if not isinstance(record, dict) or not record.get('review_id'):
                continue
            record = dict(record)
            record['airline_slug'] = slug
            # Globally unique review IDs on AirlineRatings; keep source as part of key.
            combined[(slug, str(record['review_id']))] = record

    rows = list(combined.values())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(COMBINED_JSON, rows)
    df = pd.DataFrame(rows)
    if not df.empty:
        categories = sorted({k for r in rows for k in (r.get('category_ratings') or {})})
        for category in categories:
            col = 'rating_' + re.sub(r'[^a-z0-9]+', '_', category.lower()).strip('_')
            df[col] = [(r.get('category_ratings') or {}).get(category) for r in rows]
        df['category_ratings'] = df['category_ratings'].map(
            lambda x: json.dumps(x or {}, ensure_ascii=False)
        )
    write_csv_atomic(COMBINED_CSV, df)
    print(f'Combined output: {len(rows)} reviews from '
          f'{df["airline_slug"].nunique() if not df.empty else 0} airlines -> {COMBINED_CSV}', flush=True)


def main():
    """Run selected airlines in a shared browser and rebuild combined outputs."""
    slugs = [airline_slug(url) for url in AIRLINE_URLS]
    if len(slugs) != len(set(slugs)):
        raise ValueError('AIRLINE_URLS contains the same airline more than once')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter()  # Shared across all airlines
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=HEADLESS)
            try:
                context = browser.new_context(user_agent=USER_AGENT, locale='en-US')
                for index, url in enumerate(AIRLINE_URLS, 1):
                    slug = airline_slug(url)
                    print(f'\n===== Airline {index}/{len(AIRLINE_URLS)}: {slug} =====', flush=True)
                    try:
                        scrape_airline(context, limiter, url)
                    except Exception as exc:
                        print(f'Airline {slug} failed: {exc}', flush=True)
                        traceback.print_exc()
                    finally:
                        save_combined()
            finally:
                browser.close()
    finally:
        save_combined()


if __name__ == '__main__':
    main()
