# project_ppol564_fall2021

**Socioeconomic and Geographic Factors and the Relationship to Hospital Pricing**

Ryan Ripper — Data Science I: Foundations (PPOL564-01), Fall 2021, Georgetown University

Can the price a California hospital charges for a common procedure be predicted from the income and location of the ZIP Code it sits in? This project scrapes and assembles the data to find out, then runs a machine learning pipeline to answer it.

## Background

CMS price transparency regulations, introduced by executive order in October 2020 and effective at the start of CY 2021, require every U.S. hospital to publish its procedural pricing. The stated purpose is to empower the patient-as-shopper. Existing research has largely studied the *effect of the mandate itself* on prices — not how prices are set in the first place.

This project asks two questions:

1. Do hospitals factor income into what they charge for a specific service?
2. Does the cost of a service vary across locales with differing or similar demographics?

California is the study population — 363 hospitals in the OSHPD data — chosen as a reasonable proxy for national variation, since the state spans rural to highly urbanized on nearly every measure.

## Data

| Source | What it provides |
| --- | --- |
| [U.S. Census Bureau ACS](https://www.census.gov/programs-surveys/acs) | Table S1901, "Income in the Past 12 Months (2019 inflation-adjusted dollars)" — mean and median household income by ZIP Code (ZCTA) |
| [CA.gov County and ZIP Code References](https://data.ca.gov/dataset/county-and-zip-code-references) | Authoritative list of 2,664 California ZIP Codes, intersected with the 33,120 ZCTAs in the ACS extract to leave 1,763 usable California ZIP Codes |
| [American Hospital Directory](https://www.ahd.com/) | Scraped list of California hospitals — name, bed count, city, and website. 408 with beds, 70 without |
| [CA HCAi (formerly OSHPD) Hospital Chargemasters](https://hcai.ca.gov/data-and-reports/cost-transparency/hospital-chargemasters/) | Per-hospital chargemaster files listing every billable item and service, CY 2019 |

**Outcome variable:** the line charge for **HCPCS 70450**, a computed tomography (CT) of the head or brain.

The original plan was to use HCPCS 45380 (colonoscopy with biopsy), but a colonoscopy is billed with ancillary charges — anesthesiology and so on — while the HCAi data is published per individual billable item. 70450 was chosen instead precisely because it has no add-on charges. CY 2019 was chosen to align with the ACS extract.

**Why HCAi and not the hospitals directly:** scraping each hospital's own transparency files proved impractical. Hospitals publish machine-readable pricing in wildly different formats and locations, so automating download-and-parse across hundreds of sites would have required manual intervention at nearly every step. HCAi centralizes the same information in a per-hospital directory structure that *can* be walked programmatically. Scraping AHD had its own obstacle — the site limits automation to protect its subscription model, and the report describes deceiving it with randomized wait times, though the committed `scrap_AHD()` issues a single plain `requests.get`.

**Final modeling dataset** (`data/Hospital_Data/Final_Pricing.csv`): hospital name, ZIP Code, line charge for 70450, and mean and median household income for that ZIP Code. Hospital websites were collected during scraping but are not carried into the model.

ZIP Codes reach the pricing data through `data/Hospital_Data/OSHPD_ZipCode_list.csv`. Neither AHD nor the OSHPD chargemaster files carry a ZIP Code, so the hospital-to-ZIP mapping was researched by hand. That manual work is committed as `data/Hospital_Data/OSHPD_ZipCode_lookup.csv`, and `data/build_oshpd_zipcode_list.py` performs the join, so the file is regenerable from a fresh clone:

```
OSHPD_list.csv  +  OSHPD_ZipCode_lookup.csv  →  OSHPD_ZipCode_list.csv
```

## Method

`data/Data_Collection.ipynb` handles collection and assembly:

- `scrap_AHD()` — BeautifulSoup scrape of the American Hospital Directory listing for California
- `read_OSHPD(base_code)` — walks the per-hospital OSHPD directories, finds the row matching a given HCPCS code, and coalesces across the several differently-named price columns hospitals use (`PRICE`, `June 2019 Prices`, `Charge Amount`, `STD AMOUNT`, …), filling right-to-left
- Joins hospital pricing to ACS income on ZIP Code, writing `data/Hospital_Data/Final_Pricing.csv`

`analysis/Data_Analysis.ipynb` handles modeling:

- Missingness inspection with `missingno`
- Log transformation of mean and median income (both are right-skewed; the logs are symmetric, so no bucketing into categoricals was needed)
- ZIP Code one-hot encoded as a categorical, with 90015 as the reference level
- 75/25 train/test split (`random_state=202112`), then a `GridSearchCV` pipeline — `MinMaxScaler` preprocessing feeding a model slot — over five estimators: **LinearRegression**, **BaggingRegressor**, **KNeighborsRegressor**, **DecisionTreeRegressor**, and **RandomForestRegressor** (all seeded with `random_state=202112`), cross-validated with 5-fold `KFold` and selected on mean squared error

## Results

**No model had any predictive power — all five estimators scored within a rounding error of one another, i.e., none did better than predicting the mean.** With every estimator now seeded (`random_state=202112`), the grid search reproducibly selects a depth-2 decision tree, but the "winner" is a coin flip among ties and carries no substantive meaning. (The originally reported best model, KNN with k=50, was an artifact of unseeded tree/bagging estimators; its scores were indistinguishable: CV −3,268,181, MSE 1,819,483, R² −0.04.)

| Metric | Value |
| --- | --- |
| Mean out-of-sample (CV) score | −3,268,225.54 |
| Mean squared error | 1,797,816.19 |
| R² | −0.03 |

The negative R² is not a bug. In scikit-learn, R² compares a model's fit against a fit to a single constant, and with cross-validated held-out data the test mean can differ substantially from the training mean — enough that predicting the mean beats the model. The predicted-vs-observed scatter (Figure 4) does show a weak positive relationship, but the points are close to random.

**Answering the research questions:** income does not appear to have a decisive effect on what a hospital charges, and price for a service shows no relationship to the hospital's location or surrounding demographics. Hospitals plausibly set prices from internal considerations — revenue, patient volume, actual cost of care — rather than patients' ability to pay.

The main structural limitation is sample size. Most ZIP Codes in the data contain exactly one hospital (only a few, notably 92307 in San Bernardino County and 95823 in Sacramento County, hold more than two), which strips the ZIP Code dummies of most of their power. Proposed extensions: additional procedures, and hospitals beyond California — an order of magnitude more observations.

## Requirements

```bash
pip install -r requirements.txt
```

Both `openpyxl` and `xlrd` are needed: the 2019 OSHPD download is a mix of formats, with 511 `.xlsx`/`.xlsm` files and 84 legacy `.xls` files. (Note: `read_OSHPD()` deliberately preserves the original case-sensitive extension check, which skips the 6 uppercase `.XLSX` and 2 `.xlsm` files, so the parsed output matches the committed CSVs.)

The notebooks were modernized in August 2026 to run under current pandas (≥2, verified on 3.0.2): the removed `.any(1)` call, the copy-on-write-unsafe in-place column coalesce, and the bare `except` in `read_OSHPD()` were all replaced, and the analysis notebook's estimators are now seeded so the reported results reproduce exactly from a fresh clone.

To knit the report you also need R with `rmarkdown` and `knitr`.

## Usage

```bash
git clone https://github.com/ryanripper/project_ppol564_fall2021.git
cd project_ppol564_fall2021
jupyter lab
```

- **Collection** — run `data/Data_Collection.ipynb` from within `data/` (it uses relative paths like `Census_Data/...` and `Hospital_Data/...`). Note that `scrap_AHD()` hits a live third-party site; the scraped output is already committed as `data/Hospital_Data/AHD_list.csv`.

  ⚠️ **The committed CSVs are the canonical 2021 dataset.** Re-running `read_OSHPD()` on a different machine can produce a slightly different `OSHPD_list.csv`: the "keep first duplicate" rule depends on `os.walk` directory order, and current Excel engines successfully read a few workbooks the 2021 environment silently skipped (a re-run in 2026 differed for 9 of 318 hospitals). `Final_Pricing.csv` is unaffected by a notebook re-run because it is built from the committed `OSHPD_ZipCode_list.csv` — verified to reproduce byte-identically. If you *intend* to re-parse, run `build_oshpd_zipcode_list.py` afterwards and expect downstream numbers to shift slightly.
- **ZIP Code join** — if `Hospital_Data/OSHPD_list.csv` changes, regenerate the ZIP-coded version before running the analysis:
  ```bash
  cd data && python build_oshpd_zipcode_list.py
  ```
  It fails loudly rather than emitting blank ZIP Codes if a hospital is missing from `OSHPD_ZipCode_lookup.csv`.
- **Analysis** — run `analysis/Data_Analysis.ipynb`, which reads `../data/Hospital_Data/Final_Pricing.csv`.
- **Report** — `report/ripper_ryan_final_report.Rmd` knits to HTML using `report/style.css` (figure paths are relative to `report/`, so it knits from a fresh clone). A pre-rendered `report/ripper_ryan_final_report.html` is committed if you just want to read it. Note the committed report text predates the seeded re-run and still names KNN as the best model; see Results above.

## Project structure

```
.
├── proposal/
│   ├── project_proposal.ipynb            # 11/03/2021 project proposal
│   └── project_proposal.html
├── data/
│   ├── Data_Collection.ipynb             # AHD scrape, OSHPD parse, ACS join → Final_Pricing.csv
│   ├── build_oshpd_zipcode_list.py       # OSHPD_list + ZIP lookup → OSHPD_ZipCode_list.csv
│   ├── Census_Data/                      # ACS S1901 extract + metadata, CA ZIP Code list
│   └── Hospital_Data/
│       ├── AHD_list.csv                  # Scraped California hospital list
│       ├── OSHPD_list.csv                # Parsed 70450 prices by hospital
│       ├── OSHPD_ZipCode_lookup.csv      # Hand-curated HospitalName → ZipCode mapping
│       ├── OSHPD_ZipCode_list.csv        # Same as OSHPD_list, with ZIP Codes attached
│       ├── Final_Pricing.csv             # Modeling dataset
│       └── OSHPD_2019/                   # 363 per-hospital chargemaster directories
├── analysis/
│   └── Data_Analysis.ipynb               # EDA, preprocessing, 5-estimator GridSearchCV pipeline
├── presentation/
│   ├── presentation.ipynb                # 12/01/2021 in-class slides (nbconvert slideshow)
│   ├── CA_Hosps.PNG
│   └── ripper_ryan_rar164/               # Submitted bundle incl. Presentation_Movie.mp4
├── report/
│   ├── ripper_ryan_final_report.Rmd      # 12/16/2021 final report source
│   ├── ripper_ryan_final_report.html     # Rendered report
│   ├── style.css
│   ├── images/                           # Figures 1–4
│   └── ripper_ryan_rar164/               # Submitted bundle (27 MB Census CSV deduplicated; see its Data/README.md)
├── requirements.txt
└── README.md
```

## Tech stack

pandas and NumPy (wrangling), BeautifulSoup and requests (scraping), scikit-learn (pipeline, GridSearchCV, KFold, five estimators), plotnine and missingno (visualization), Jupyter, R Markdown (final report).

## Author

Ryan Ripper — Georgetown University, Fall 2021
