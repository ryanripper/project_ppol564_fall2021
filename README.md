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

ZIP Codes reach the pricing data through `data/Hospital_Data/OSHPD_ZipCode_list.csv`, which the collection notebook reads but never writes — that file was assembled outside the committed pipeline.

## Method

`data/Data_Collection.ipynb` handles collection and assembly:

- `scrap_AHD()` — BeautifulSoup scrape of the American Hospital Directory listing for California
- `read_OSHPD(base_code)` — walks the per-hospital OSHPD directories, finds the row matching a given HCPCS code, and coalesces across the several differently-named price columns hospitals use (`PRICE`, `June 2019 Prices`, `Charge Amount`, `STD AMOUNT`, …), filling right-to-left
- Joins hospital pricing to ACS income on ZIP Code, writing `data/Hospital_Data/Final_Pricing.csv`

`analysis/Data_Analysis.ipynb` handles modeling:

- Missingness inspection with `missingno`
- Log transformation of mean and median income (both are right-skewed; the logs are symmetric, so no bucketing into categoricals was needed)
- ZIP Code one-hot encoded as a categorical, with 90015 as the reference level
- 75/25 train/test split (`random_state=202112`), then a `GridSearchCV` pipeline — `MinMaxScaler` preprocessing feeding a model slot — over five estimators: **LinearRegression**, **BaggingRegressor**, **KNeighborsRegressor**, **DecisionTreeRegressor**, and **RandomForestRegressor**, cross-validated with 5-fold `KFold` and selected on mean squared error

## Results

The best-performing model was **K-nearest neighbors with k=50** — and it had no predictive power:

| Metric | Value |
| --- | --- |
| Mean out-of-sample (CV) score | −3,268,181.31 |
| Mean squared error | 1,819,483.47 |
| R² | −0.04 |

The negative R² is not a bug. In scikit-learn, R² compares a model's fit against a fit to a single constant, and with cross-validated held-out data the test mean can differ substantially from the training mean — enough that predicting the mean beats the model. The predicted-vs-observed scatter (Figure 4) does show a weak positive relationship, but the points are close to random.

**Answering the research questions:** income does not appear to have a decisive effect on what a hospital charges, and price for a service shows no relationship to the hospital's location or surrounding demographics. Hospitals plausibly set prices from internal considerations — revenue, patient volume, actual cost of care — rather than patients' ability to pay.

The main structural limitation is sample size. Most ZIP Codes in the data contain exactly one hospital (only a few, notably 92307 in San Bernardino County and 95823 in Sacramento County, hold more than two), which strips the ZIP Code dummies of most of their power. Proposed extensions: additional procedures, and hospitals beyond California — an order of magnitude more observations.

## Requirements

No `requirements.txt` is committed. To run the notebooks:

```bash
pip install pandas numpy scikit-learn beautifulsoup4 requests plotnine missingno openpyxl xlrd jupyter
```

To knit the report you also need R with `rmarkdown` and `knitr`.

## Usage

```bash
git clone https://github.com/ryanripper/project_ppol564_fall2021.git
cd project_ppol564_fall2021
jupyter lab
```

- **Collection** — run `data/Data_Collection.ipynb` from within `data/` (it uses relative paths like `Census_Data/...` and `Hospital_Data/...`). Note that `scrap_AHD()` hits a live third-party site; the scraped output is already committed as `data/Hospital_Data/AHD_list.csv`.
- **Analysis** — run `analysis/Data_Analysis.ipynb`, which reads `../data/Hospital_Data/Final_Pricing.csv`.
- **Report** — `report/ripper_ryan_final_report.Rmd` knits to HTML using `report/style.css`. The figure paths in the `.Rmd` are absolute (`/Users/ryanripper/Desktop/PPOL_564/...`) and need to be repointed at `report/images/` before it will knit elsewhere. A pre-rendered `report/ripper_ryan_final_report.html` is committed if you just want to read it.

## Project structure

```
.
├── proposal/
│   ├── project_proposal.ipynb            # 11/03/2021 project proposal
│   └── project_proposal.html
├── data/
│   ├── Data_Collection.ipynb             # AHD scrape, OSHPD parse, ACS join → Final_Pricing.csv
│   ├── Census_Data/                      # ACS S1901 extract + metadata, CA ZIP Code list
│   └── Hospital_Data/
│       ├── AHD_list.csv                  # Scraped California hospital list
│       ├── OSHPD_list.csv                # Parsed 70450 prices by hospital
│       ├── OSHPD_ZipCode_list.csv        # Same, with ZIP Codes attached
│       ├── Final_Pricing.csv             # Modeling dataset
│       └── OSHPD_2019/                   # 363 per-hospital chargemaster directories
├── analysis/
│   └── Data_Analysis.ipynb               # EDA, preprocessing, 5-estimator GridSearchCV pipeline
├── presentation/
│   ├── presentation.ipynb                # 12/01/2021 in-class slides (nbconvert slideshow)
│   ├── CA_Hosps.PNG
│   ├── ripper_ryan_rar164/               # Submitted bundle incl. Presentation_Movie.mp4
│   └── ripper_ryan_rar164.zip            # Same bundle, zipped
├── report/
│   ├── ripper_ryan_final_report.Rmd      # 12/16/2021 final report source
│   ├── ripper_ryan_final_report.html     # Rendered report
│   ├── style.css
│   ├── images/                           # Figures 1–4
│   ├── ripper_ryan_rar164/               # Submitted bundle incl. data
│   └── ripper_ryan_rar164.zip            # Same bundle, zipped
└── README.md
```

## Tech stack

pandas and NumPy (wrangling), BeautifulSoup and requests (scraping), scikit-learn (pipeline, GridSearchCV, KFold, five estimators), plotnine and missingno (visualization), Jupyter, R Markdown (final report).

## Author

Ryan Ripper — Georgetown University, Fall 2021
