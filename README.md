# Tunisian Real Estate EstateMind Scraper 🚀

EstateMind is a high-performance, automated real estate data harvesting engine designed to map the Tunisian property market.

## 🛠 Project Architecture
The system follows a 3-stage pipeline to ensure data completeness and quality:

### 1. 🔍 Crawling (Discovery Stage)
**Main Logic**: `discovery/explorer.py`
The engine recursively discovers the Tunisian real estate ecosystem.
- **Deep Discovery**: Uses SerpAPI to paginate through Google results for regional keywords (e.g., "agence immobilière Sousse").
- **Exhaustive Mapping**: Scans for `sitemap.xml` and DNS records (`crt.sh`) to find niche agencies that are normally hidden.
- **Quality Score**: Every discovered site is validated for relevance before being added to the target list.

### 2. ✂️ Scraping (Extraction Stage)
**Main Logic**: `scrapers/` (Generic & Specialized)
Once a target is validated, the scraper dives deep into the property categories.
- **Specialized Engines** (`mubawab_scraper.py`, `menzili_scraper.py`): Custom logic for major platforms.
- **Heuristic Generic Scraper** (`generic_scraper.py`): A robust fallback system that automatically extracts titles, prices, and images from any real estate site using BS4.
- **Data Normalization**: Cleans and validates prices, surfaces, and locations to ensure a uniform dataset.

### 3. 💾 Data Storage (MongoDB Integration)
**Main Logic**: `database/models.py` & `processing/reporting.py`
All extracted data is synced in real-time to a Cloud database.
- **MongoDB Atlas**: Fully persistent storage allowing for complex queries and analysis.
- **Deduplication**: Uses data hashing to ensure no property is ever saved twice, even across multiple runs.
- **Excel Export**: For every cycle, a fresh `final_listings_report.xlsx` is generated for instant business use.

## 🛠 Command Reference

### 1. The Main Engine (Collection + Scraping + AI Sync)
Run this to start a fresh 24h cycle of data gathering. It will automatically update your AI model at the end.
```powershell
python main.py
```

### 2. The AI Pipeline (Clean + Encode + Retrain)
Run this if you only want to process existing data in MongoDB and update your model without a new scrape.
```powershell
python run_pipeline.py
```

### 3. Model Testing
Test your current AI model with a simulated property.
```powershell
python predict_example.py
```

### 4. Database Check
Quickly see how many listings you have in your local environment.
```powershell
python get_count.py
```

## 📂 Data Structure
- `pipeline/`: Standardized CSVs for training.
- `models/`: Your actual AI brain (`.joblib`) and performance history.
- `downloads/images/`: Automatically collected media folders.
