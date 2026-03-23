"""
scheduler.py – Run the scraper on a schedule (daily incremental updates).

Usage:
    python scheduler.py               # run now, then every 24 h
    python scheduler.py --interval 12 # run every 12 hours
"""
import argparse
import logging
import time
from scraper import scrape

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=24.0,
                        help="Hours between runs (default 24)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay",   type=float, default=1.2)
    args = parser.parse_args()

    while True:
        log.info("▶ Starting scheduled scrape run…")
        try:
            scrape(workers=args.workers, delay=args.delay)
        except Exception as exc:
            log.error("Scrape run failed: %s", exc)
        next_run = args.interval * 3600
        log.info("Next run in %.0f hours. Sleeping…", args.interval)
        time.sleep(next_run)


if __name__ == "__main__":
    main()
