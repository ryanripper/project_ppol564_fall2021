"""
Build Hospital_Data/OSHPD_ZipCode_list.csv.

Background
----------
Data_Collection.ipynb parses the OSHPD chargemaster directories into
Hospital_Data/OSHPD_list.csv (HospitalName, HospitalPrice_70450), and later
reads Hospital_Data/OSHPD_ZipCode_list.csv to join hospital pricing against
Census income by ZIP Code. Nothing in the notebook ever *wrote* that second
file -- the ZIP Codes were researched by hand and pasted in, which left a hole
in the middle of the pipeline: a fresh clone could not regenerate the file.

This script closes that hole. The manual research is preserved as a committed
lookup table, Hospital_Data/OSHPD_ZipCode_lookup.csv, and the join that
produces OSHPD_ZipCode_list.csv is now reproducible code.

    OSHPD_list.csv  +  OSHPD_ZipCode_lookup.csv  ->  OSHPD_ZipCode_list.csv

Usage
-----
Run from the data/ directory:

    python build_oshpd_zipcode_list.py

Maintaining the lookup
----------------------
OSHPD_ZipCode_lookup.csv is a hand-curated HospitalName -> ZipCode mapping.
When new hospitals appear in OSHPD_list.csv, this script will report them as
missing and exit non-zero rather than silently writing rows with a blank ZIP
Code. Add the new hospitals to the lookup and re-run.
"""

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "Hospital_Data"

OSHPD_LIST = DATA_DIR / "OSHPD_list.csv"
ZIPCODE_LOOKUP = DATA_DIR / "OSHPD_ZipCode_lookup.csv"
OUTPUT = DATA_DIR / "OSHPD_ZipCode_list.csv"


def build():
    """Join OSHPD pricing to the curated ZIP Code lookup and write the result."""

    # Pricing parsed out of the OSHPD chargemaster files by Data_Collection.ipynb.
    # Column 0 is the original unnamed index; keep it so the output matches the
    # file the notebook has always consumed.
    oshpd = pd.read_csv(OSHPD_LIST, index_col=0)

    # Hand-curated hospital -> ZIP Code mapping. Read ZipCode as a string so
    # leading zeros survive (harmless for California, but correct in general).
    lookup = pd.read_csv(ZIPCODE_LOOKUP, dtype={"ZipCode": str})

    duplicates = lookup.loc[lookup.HospitalName.duplicated(), "HospitalName"]
    if not duplicates.empty:
        print(
            f"ERROR: {ZIPCODE_LOOKUP.name} has duplicate hospital names:",
            file=sys.stderr,
        )
        for name in sorted(duplicates.unique()):
            print(f"  - {name}", file=sys.stderr)
        return 1

    merged = oshpd.merge(lookup, on="HospitalName", how="left")

    missing = merged.loc[merged.ZipCode.isna(), "HospitalName"]
    if not missing.empty:
        print(
            f"ERROR: {len(missing)} hospital(s) in {OSHPD_LIST.name} have no ZIP "
            f"Code in {ZIPCODE_LOOKUP.name}:",
            file=sys.stderr,
        )
        for name in sorted(missing):
            print(f"  - {name}", file=sys.stderr)
        print(
            "\nAdd them to the lookup (HospitalName,ZipCode) and re-run.",
            file=sys.stderr,
        )
        return 1

    unused = set(lookup.HospitalName) - set(oshpd.HospitalName)
    if unused:
        print(
            f"Note: {len(unused)} lookup entries do not appear in "
            f"{OSHPD_LIST.name} and were ignored."
        )

    # Restore the original index so the output is a drop-in replacement.
    merged.index = oshpd.index
    merged.to_csv(OUTPUT)

    print(f"Wrote {OUTPUT} ({len(merged)} hospitals).")
    return 0


if __name__ == "__main__":
    sys.exit(build())
