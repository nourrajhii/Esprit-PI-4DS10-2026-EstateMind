"""
==============================================================================
EstateMind AI Pipeline Orchestrator
==============================================================================
Usage:
    python run_pipeline.py

This script runs all ML pipeline stages in order:
  Stage 0: Sync latest data from MongoDB (runs a scraper cycle if requested)
  Stage 1: Clean raw listings (cleaner.py)
  Stage 2: Feature engineering (feature_engineering.py)
  Stage 3: Print a final feature audit report
==============================================================================
"""

import asyncio
import logging
import os
import sys
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PIPELINE_DIR = "pipeline"


async def run():
    os.makedirs(PIPELINE_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("      EstateMind AI Data Pipeline")
    logger.info("=" * 60)

    # ── Stage 1: Clean ────────────────────────────────────────────────────
    from database.models import init_db
    from processing.cleaner import run_cleaning_pipeline

    await init_db()
    n_clean = await run_cleaning_pipeline()

    if n_clean < 50:
        logger.error(
            f"Only {n_clean} rows passed cleaning — too few for training. "
            "Run a scraping cycle first with `python main.py`."
        )
        sys.exit(1)

    # ── Stage 2: Feature Engineering ──────────────────────────────────────
    from processing.feature_engineering import run_feature_engineering

    result = run_feature_engineering()

    # ── Stage 3: Final Feature Audit ──────────────────────────────────────
    logger.info("─" * 60)
    logger.info("STAGE 3 │ Feature Audit Report")

    report = pd.read_csv(f"{PIPELINE_DIR}/feature_report.csv", index_col=0)
    important_cols = ["mean", "std", "min", "max", "missing_%"]
    visible = [c for c in important_cols if c in report.columns]
    logger.info(f"\n{report[visible].to_string()}")

    logger.info("─" * 60)
    logger.info("PIPELINE COMPLETE ✅")
    logger.info(f"  Features   : {result['n_features']}")
    logger.info(f"  Train rows : {result['n_train']}")
    logger.info(f"  Test rows  : {result['n_test']}")
    logger.info("─" * 60)
    logger.info("Output files in pipeline/:")
    for fname in sorted(os.listdir(PIPELINE_DIR)):
        fpath = os.path.join(PIPELINE_DIR, fname)
        size  = os.path.getsize(fpath) / 1024
        logger.info(f"  📄 {fname:35s} {size:8.1f} KB")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run())
