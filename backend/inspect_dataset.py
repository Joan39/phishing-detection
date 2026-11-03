import pandas as pd
from pathlib import Path

# Step 1: Specify the dataset filename
file_path = Path("original") / "PhiUSIIL_Phishing_URL_Dataset.csv"

# Step 2: Check whether the dataset exists
if not file_path.exists():
    print("Dataset not found!")
    print("Make sure the CSV file is in the same folder as this script.")
    print("Check the exact filename in VS Code.")
else:
    # Step 3: Load the dataset
    df = pd.read_csv(file_path)

    # Step 4: Display the dataset size
    print("\n--- DATASET SIZE ---")
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    # Step 5: Display the first five records
    print("\n--- FIRST FIVE RECORDS ---")
    print(df.head())

    # Step 6: Display all column names
    print("\n--- COLUMN NAMES ---")
    print(df.columns.tolist())

    # Step 7: Check for missing values
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    # Step 8: Count duplicate records
    print("\n--- DUPLICATE RECORDS ---")
    print("Number of duplicate rows:", df.duplicated().sum())

    # Step 9: Display data types
    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    # Step 10: Examine the dataset summary
    print("\n--- DATASET SUMMARY ---")
    print(df.info())

    print("\nDataset inspection completed successfully!")
    
# STEP 1: Check the target labels

print("Unique labels:")
print(df["label"].unique())

print("\nNumber of records per label:")
label_counts = df["label"].value_counts().sort_index()
print(label_counts)

print("\nPercentage of records per label:")
label_percentages = (
    df["label"].value_counts(normalize=True).sort_index() * 100
)
print(label_percentages.round(2))

# STEP 2: Check duplicate URLs and domains

print("\n--- DUPLICATE URL CHECK ---")
duplicate_urls = df["URL"].duplicated().sum()
print("Number of duplicate URLs:", duplicate_urls)

print("\n--- DUPLICATE DOMAIN CHECK ---")
duplicate_domains = df["Domain"].duplicated().sum()
print("Number of duplicate domains:", duplicate_domains)

# Check whether identical URLs have conflicting labels
url_label_counts = df.groupby("URL")["label"].nunique()
conflicting_urls = (url_label_counts > 1).sum()

print("\n--- URL LABEL CONSISTENCY CHECK ---")
print("URLs associated with conflicting labels:", conflicting_urls)

# STEP 3: Inspect duplicate URLs

# Count how many rows belong to repeated URLs
repeated_url_rows = df[df["URL"].duplicated(keep=False)]

print("\n--- DUPLICATE URL DETAILS ---")
print("Rows involved in repeated URLs:", len(repeated_url_rows))

# Display examples of repeated URLs
print("\nExamples of repeated URLs:")
print(
    repeated_url_rows[
        ["URL", "Domain", "label"]
    ].sort_values("URL").head(20).to_string(index=False)
)

# Check whether repeated URL rows have identical features
duplicate_url_groups = df.groupby("URL", sort=False)

print("\n--- REPEATED URL GROUPS ---")
print(
    "Number of URL groups appearing more than once:",
    (duplicate_url_groups.size() > 1).sum()
)


# STEP 4: Check whether repeated URLs have identical features

# Exclude FILENAME because it is an identifier
comparison_columns = [
    column for column in df.columns
    if column != "FILENAME"
]

# Get all rows belonging to repeated URL groups
repeated_rows = df[
    df["URL"].duplicated(keep=False)
].copy()

# Count distinct records within each repeated URL group
distinct_records = (
    repeated_rows[comparison_columns]
    .drop_duplicates()
)

print("\n--- DUPLICATE FEATURE CHECK ---")
print("Rows in repeated URL groups:", len(repeated_rows))
print("Distinct records after excluding FILENAME:",
      len(distinct_records))

# Check for exact duplicate feature records
exact_duplicate_features = df[
    comparison_columns
].duplicated().sum()

print("Duplicate feature records excluding FILENAME:",
      exact_duplicate_features)

# STEP 5: Verify features of repeated URLs

# Find all rows belonging to repeated URLs
repeated_rows = df[
    df["URL"].duplicated(keep=False)
].copy()

# Compare records without the filename
comparison_columns = [
    col for col in df.columns
    if col not in ["FILENAME"]
]

# Count distinct feature records for each URL
feature_counts = (
    repeated_rows.groupby("URL")[comparison_columns]
    .apply(lambda group: len(group.drop_duplicates()))
)

print("\n--- REPEATED URL FEATURE VERIFICATION ---")
print("Repeated URL groups:", len(feature_counts))
print("Groups with identical feature records:",
      (feature_counts == 1).sum())
print("Groups with different feature records:",
      (feature_counts > 1).sum())

# Display examples where repeated URLs have different features
different_feature_urls = feature_counts[feature_counts > 1].index

if len(different_feature_urls) > 0:
    print("\nExamples with different feature records:")
    print(
        repeated_rows[
            repeated_rows["URL"].isin(different_feature_urls)
        ][comparison_columns].head(10).to_string(index=False)
    )
else:
    print("\nAll repeated URL groups have identical feature records.")

# STEP 6: Inspect one repeated URL in detail

# Find URLs that appear more than once
url_counts = df["URL"].value_counts()
repeated_urls = url_counts[url_counts > 1].index

# Select one repeated URL
if len(repeated_urls) > 0:
    sample_url = repeated_urls[0]

    print("\n--- SELECTED REPEATED URL ---")
    print(sample_url)

    # Display all records for that URL
    sample_records = df[df["URL"] == sample_url]

    print("\n--- RECORDS FOR THIS URL ---")
    print(sample_records.T.to_string())

    # Compare all columns except the filename
    comparison_columns = [
        col for col in df.columns if col != "FILENAME"
    ]

    print("\n--- IDENTICAL FEATURE CHECK ---")
    print(
        "Are the records identical apart from FILENAME?",
        sample_records[comparison_columns]
        .drop_duplicates()
        .shape[0] == 1
    )
else:
    print("No repeated URLs were found.")

# STEP 7: Check whether URL-based features are consistent

# Select one repeated URL for comparison
sample_url = "https://att-104164.weeblysite.com/"

sample_records = df[df["URL"] == sample_url].copy()

# Features that can be calculated directly from the URL
print("\n--- URL FEATURE CONSISTENCY CHECK ---")
print("URL:", sample_url)
print("Number of records:", len(sample_records))
print("Actual URL string length:", len(sample_url))

print("\nRecorded URL lengths:")
print(sample_records["URLLength"].tolist())

print("\nRecorded letter counts:")
print(sample_records["NoOfLettersInURL"].tolist())

print("\nRecorded digit counts:")
print(sample_records["NoOfDegitsInURL"].tolist())

print("\nRecorded labels:")
print(sample_records["label"].tolist())

# Count URLs whose recorded URLLength differs from the actual string length
actual_lengths = df["URL"].str.len()
length_mismatches = (df["URLLength"] != actual_lengths).sum()

print("\n--- FULL DATASET LENGTH CHECK ---")
print("Records where URLLength differs from actual URL length:",
      length_mismatches)
print("Total records checked:", len(df))

# STEP 8: Inspect URL length mismatches

actual_lengths = df["URL"].str.len()

mismatched_records = df.loc[
    df["URLLength"] != actual_lengths,
    ["URL", "URLLength", "label"]
].copy()

mismatched_records["ActualURLLength"] = (
    mismatched_records["URL"].str.len()
)

mismatched_records["LengthDifference"] = (
    mismatched_records["URLLength"]
    - mismatched_records["ActualURLLength"]
)

print("\n--- FIVE URL LENGTH MISMATCHES ---")
print(mismatched_records.head(5).to_string(index=False))

print("\n--- LENGTH DIFFERENCE SUMMARY ---")
print(mismatched_records["LengthDifference"].describe())

print("\n--- MOST COMMON LENGTH DIFFERENCES ---")
print(mismatched_records["LengthDifference"].value_counts().head(10))

# STEP 9: Check other URL-related feature patterns

import pandas as pd

# Calculate actual URL character counts
actual_url_length = df["URL"].str.len()
actual_letters = df["URL"].str.count(r"[A-Za-z]")
actual_digits = df["URL"].str.count(r"\d")

print("\n--- URL FEATURE COMPARISON ---")

print(
    "URLLength mismatches:",
    (df["URLLength"] != actual_url_length).sum()
)

print(
    "Letter-count mismatches:",
    (df["NoOfLettersInURL"] != actual_letters).sum()
)

print(
    "Digit-count mismatches:",
    (df["NoOfDegitsInURL"] != actual_digits).sum()
)

print("\n--- EXAMPLES WITH LETTER-COUNT MISMATCHES ---")

letter_mismatches = df.loc[
    df["NoOfLettersInURL"] != actual_letters,
    ["URL", "NoOfLettersInURL", "label"]
].copy()

letter_mismatches["ActualLetters"] = actual_letters[
    letter_mismatches.index
]

print(letter_mismatches.head(5).to_string(index=False))

print("\n--- EXAMPLES WITH DIGIT-COUNT MISMATCHES ---")

digit_mismatches = df.loc[
    df["NoOfDegitsInURL"] != actual_digits,
    ["URL", "NoOfDegitsInURL", "label"]
].copy()

digit_mismatches["ActualDigits"] = actual_digits[
    digit_mismatches.index
]

print(digit_mismatches.head(5).to_string(index=False))

# STEP 10: Investigate URL feature calculation patterns

import re

# Select a few records to investigate
sample = df.head(10).copy()

# Calculate alternative counts
sample["ActualURLLength"] = sample["URL"].str.len()

sample["LettersInFullURL"] = sample["URL"].str.count(r"[A-Za-z]")

sample["LettersInDomain"] = sample["Domain"].str.count(r"[A-Za-z]")

sample["LettersWithoutProtocol"] = (
    sample["URL"]
    .str.replace(r"^https?://", "", regex=True)
    .str.count(r"[A-Za-z]")
)

sample["DigitsInFullURL"] = sample["URL"].str.count(r"\d")

sample["ActualDomainLength"] = sample["Domain"].str.len()

print("\n--- URL FEATURE INVESTIGATION ---")

print(
    sample[
        [
            "URL",
            "Domain",
            "URLLength",
            "ActualURLLength",
            "NoOfLettersInURL",
            "LettersInFullURL",
            "LettersInDomain",
            "LettersWithoutProtocol",
            "NoOfDegitsInURL",
            "DigitsInFullURL",
            "DomainLength",
            "ActualDomainLength",
        ]
    ].to_string(index=False)
)

print("\n--- DOMAIN LENGTH CHECK ---")

domain_length_mismatches = (
    df["DomainLength"] != df["Domain"].str.len()
).sum()

print("Domain length mismatches:", domain_length_mismatches)
print("Total records:", len(df))

# STEP 11: Investigate URL ratio calculations

print("\n--- CHECKING URL RATIOS ---")

# Avoid division by zero
safe_url_length = df["URLLength"].replace(0, pd.NA)

# Calculate ratios using the recorded URL length
calculated_letter_ratio = (
    df["NoOfLettersInURL"] / safe_url_length
)

calculated_digit_ratio = (
    df["NoOfDegitsInURL"] / safe_url_length
)

# Compare calculated ratios with recorded ratios
letter_ratio_matches = (
    (calculated_letter_ratio - df["LetterRatioInURL"]).abs() < 0.002
)

digit_ratio_matches = (
    (calculated_digit_ratio - df["DegitRatioInURL"]).abs() < 0.002
)

print(
    "Letter ratio matches using recorded URLLength:",
    letter_ratio_matches.sum(),
    "out of",
    len(df)
)

print(
    "Digit ratio matches using recorded URLLength:",
    digit_ratio_matches.sum(),
    "out of",
    len(df)
)

# Show examples where the calculated ratios differ
print("\n--- LETTER RATIO EXAMPLES ---")

letter_check = df.loc[
    ~letter_ratio_matches,
    [
        "URL",
        "URLLength",
        "NoOfLettersInURL",
        "LetterRatioInURL",
    ]
].copy()

letter_check["CalculatedLetterRatio"] = calculated_letter_ratio[
    letter_check.index
]

print(letter_check.head(5).to_string(index=False))

print("\n--- DIGIT RATIO EXAMPLES ---")

digit_check = df.loc[
    ~digit_ratio_matches,
    [
        "URL",
        "URLLength",
        "NoOfDegitsInURL",
        "DegitRatioInURL",
    ]
].copy()

digit_check["CalculatedDigitRatio"] = calculated_digit_ratio[
    digit_check.index
]

print(digit_check.head(5).to_string(index=False))

# STEP 12: Inspect the target labels and feature ranges

print("\n--- TARGET LABEL MEANING ---")
print("Label 0 = Phishing")
print("Label 1 = Legitimate")

print("\n--- LABEL COUNTS ---")
print(df["label"].value_counts().sort_index())

print("\n--- LABEL PERCENTAGES ---")
print(
    (df["label"].value_counts(normalize=True).sort_index() * 100)
    .round(2)
)

print("\n--- NUMERIC FEATURE SUMMARY ---")
print(df.describe().T.to_string())

print("\n--- LABEL VALUE VALIDATION ---")
print("Unique labels:", sorted(df["label"].unique()))
print("Unexpected labels:", sorted(set(df["label"].unique()) - {0, 1}))

print("\n--- BINARY FEATURE VALIDATION ---")

binary_columns = [
    "IsDomainIP",
    "HasObfuscation",
    "IsHTTPS",
    "HasTitle",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "HasDescription",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "label"
]

for column in binary_columns:
    values = sorted(df[column].dropna().unique().tolist())
    if not set(values).issubset({0, 1}):
        print(f"{column}: unexpected values {values}")

print("Binary feature validation completed.")

# STEP 13: Investigate extreme numerical values
# This step does not change the original dataset.

extreme_columns = [
    "URLLength",
    "LineOfCode",
    "LargestLineLength",
    "NoOfPopup",
    "NoOfiFrame",
    "NoOfCSS",
    "NoOfSelfRef",
    "NoOfExternalRef"
]

for column in extreme_columns:
    print(f"\n--- TOP 5 RECORDS FOR {column} ---")

    top_records = df.nlargest(5, column)[
        ["URL", "Domain", column, "label"]
    ]

    print(top_records.to_string(index=False))

print("\n--- ZERO VALUE COUNTS ---")

for column in extreme_columns:
    zero_count = (df[column] == 0).sum()
    percentage = zero_count / len(df) * 100

    print(
        f"{column}: {zero_count:,} records "
        f"({percentage:.2f}%) have a value of zero"
    )

# STEP 14: Review the dataset's feature definitions

print("DATASET OVERVIEW")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nCOLUMN NAMES")
for number, column in enumerate(df.columns, start=1):
    print(f"{number}. {column}")

print("\nFEATURES THAT NEED DEFINITION CHECKS")
features_to_review = [
    "URLLength",
    "NoOfLettersInURL",
    "NoOfDegitsInURL",
    "LetterRatioInURL",
    "DegitRatioInURL",
    "URLSimilarityIndex",
    "TLDLegitimateProb",
    "URLCharProb",
    "LineOfCode",
    "LargestLineLength",
    "NoOfPopup",
    "NoOfiFrame",
    "NoOfCSS",
    "NoOfSelfRef",
    "NoOfExternalRef"
]

for feature in features_to_review:
    if feature in df.columns:
        print(f"\n{feature}")
        print("Data type:", df[feature].dtype)
        print("Unique values:", df[feature].nunique())
        print("Minimum:", df[feature].min())
        print("Maximum:", df[feature].max())

# STEP 15: Inspect unusual records and repeated URLs

import pandas as pd

# 1. Inspect the five longest URLs without printing the full URL contents
print("TOP 5 URL LENGTH RECORDS")
print(
    df.nlargest(5, "URLLength")[
        ["Domain", "URLLength", "label"]
    ].to_string(index=False)
)

# 2. Inspect the five highest LargestLineLength values
print("\nTOP 5 LARGEST LINE LENGTH RECORDS")
print(
    df.nlargest(5, "LargestLineLength")[
        ["Domain", "LargestLineLength", "LineOfCode", "label"]
    ].to_string(index=False)
)

# 3. Inspect repeated URLs
print("\nREPEATED URL SUMMARY")

repeated_url_mask = df["URL"].duplicated(keep=False)

print("Rows belonging to repeated URLs:", repeated_url_mask.sum())
print("Number of distinct repeated URLs:",
      df.loc[repeated_url_mask, "URL"].nunique())

# 4. Check whether repeated URLs have conflicting labels
repeated_url_label_counts = (
    df.loc[repeated_url_mask]
      .groupby("URL")["label"]
      .nunique()
)

print(
    "Repeated URLs with conflicting labels:",
    (repeated_url_label_counts > 1).sum()
)

# 5. Inspect the feature values of the most extreme records
print("\nEXTREME RECORD FEATURE COMPARISON")

extreme_records = df.nlargest(5, "LargestLineLength")

features_to_display = [
    "Domain",
    "URLLength",
    "LineOfCode",
    "LargestLineLength",
    "NoOfPopup",
    "NoOfiFrame",
    "NoOfCSS",
    "NoOfJS",
    "label"
]

print(extreme_records[features_to_display].to_string(index=False))

# STEP 16: Check repeated URLs and domains

print("1. REPEATED URL SUMMARY")

repeated_mask = df["URL"].duplicated(keep=False)

print("Rows belonging to repeated URLs:", repeated_mask.sum())
print(
    "Distinct repeated URLs:",
    df.loc[repeated_mask, "URL"].nunique()
)

print("\n2. LABEL CONSISTENCY FOR REPEATED URLs")

if repeated_mask.any():
    label_counts = (
        df.loc[repeated_mask]
        .groupby("URL")["label"]
        .nunique()
    )

    print(
        "Repeated URLs with conflicting labels:",
        (label_counts > 1).sum()
    )
else:
    print("No repeated URLs found.")

print("\n3. REPEATED DOMAIN SUMMARY")

repeated_domain_mask = df["Domain"].duplicated(keep=False)

print(
    "Rows belonging to repeated domains:",
    repeated_domain_mask.sum()
)
print(
    "Distinct repeated domains:",
    df.loc[repeated_domain_mask, "Domain"].nunique()
)

print("\n4. EXAMPLE REPEATED DOMAINS")

domain_counts = df["Domain"].value_counts()

print(
    domain_counts[domain_counts > 1]
    .head(10)
    .to_string()
)

print("\n5. LABEL COUNTS WITHIN THE MOST REPEATED DOMAINS")

top_domains = domain_counts[domain_counts > 1].head(5).index

print(
    df[df["Domain"].isin(top_domains)]
    .groupby("Domain")["label"]
    .value_counts()
    .unstack(fill_value=0)
    .to_string()
)

# STEP 17: Examine label distribution by domain

print("1. LABEL DISTRIBUTION ACROSS ALL RECORDS")

print(
    df["label"]
    .value_counts()
    .rename(index={0: "Phishing", 1: "Legitimate"})
    .to_string()
)

print("\n2. DOMAINS WITH BOTH LABELS")

labels_per_domain = df.groupby("Domain")["label"].nunique()

mixed_domains = labels_per_domain[labels_per_domain > 1]

print("Total distinct domains:", df["Domain"].nunique())
print("Domains containing both labels:", len(mixed_domains))

print("\n3. EXAMPLES OF DOMAINS WITH BOTH LABELS")

if len(mixed_domains) > 0:
    mixed_domain_examples = (
        df[df["Domain"].isin(mixed_domains.index)]
        .groupby("Domain")["label"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={0: "Phishing", 1: "Legitimate"})
        .sort_values(
            by=["Phishing", "Legitimate"],
            ascending=False
        )
    )

    print(mixed_domain_examples.head(15).to_string())
else:
    print("No domains contain both labels.")

print("\n4. MOST REPEATED DOMAINS AND THEIR LABELS")

domain_label_counts = (
    df.groupby(["Domain", "label"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "Phishing", 1: "Legitimate"})
)

domain_label_counts["Total"] = (
    domain_label_counts["Phishing"]
    + domain_label_counts["Legitimate"]
)

print(
    domain_label_counts
    .sort_values("Total", ascending=False)
    .head(15)
    .to_string()
)

# STEP 17: Examine label distribution by domain

print("1. LABEL DISTRIBUTION ACROSS ALL RECORDS")

print(
    df["label"]
    .value_counts()
    .rename(index={0: "Phishing", 1: "Legitimate"})
    .to_string()
)

print("\n2. DOMAINS WITH BOTH LABELS")

labels_per_domain = df.groupby("Domain")["label"].nunique()
mixed_domains = labels_per_domain[labels_per_domain > 1]

print("Total distinct domains:", df["Domain"].nunique())
print("Domains containing both labels:", len(mixed_domains))

print("\n3. EXAMPLES OF DOMAINS WITH BOTH LABELS")

if len(mixed_domains) > 0:
    mixed_domain_examples = (
        df[df["Domain"].isin(mixed_domains.index)]
        .groupby("Domain")["label"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={0: "Phishing", 1: "Legitimate"})
        .sort_values(
            by=["Phishing", "Legitimate"],
            ascending=False
        )
    )

    print(mixed_domain_examples.head(15).to_string())
else:
    print("No domains contain both labels.")

print("\n4. MOST REPEATED DOMAINS AND THEIR LABELS")

domain_label_counts = (
    df.groupby(["Domain", "label"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "Phishing", 1: "Legitimate"})
)

domain_label_counts["Total"] = (
    domain_label_counts["Phishing"]
    + domain_label_counts["Legitimate"]
)

print(
    domain_label_counts
    .sort_values("Total", ascending=False)
    .head(15)
    .to_string()
)

# STEP 18: Inspect potentially leakage-prone features

features_to_check = [
    "URLSimilarityIndex",
    "TLDLegitimateProb",
    "URLCharProb"
]

print("FEATURE CORRELATION WITH THE LABEL")
print("=" * 50)

correlations = df[features_to_check + ["label"]].corr(numeric_only=True)

print(correlations["label"].sort_values(ascending=False).to_string())


print("\nFEATURE VALUES BY CLASS")
print("=" * 50)

for feature in features_to_check:
    print(f"\n{feature}")

    print(
        df.groupby("label")[feature]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .rename(index={0: "Phishing", 1: "Legitimate"})
        .round(4)
        .to_string()
    )


print("\nMISSING VALUES IN THESE FEATURES")
print("=" * 50)

print(df[features_to_check].isnull().sum().to_string())
# STEP 19: Audit raw URL structure by class

from urllib.parse import urlparse
import pandas as pd

audit = df[["URL", "label"]].copy()

audit["starts_with_https"] = audit["URL"].str.lower().str.startswith("https://")
audit["starts_with_www"] = audit["URL"].str.lower().apply(
    lambda url: urlparse(url).netloc.startswith("www.")
)
audit["has_path"] = audit["URL"].apply(
    lambda url: bool(urlparse(url).path.strip("/"))
)
audit["has_query"] = audit["URL"].apply(
    lambda url: bool(urlparse(url).query)
)
audit["has_at_symbol"] = audit["URL"].str.contains("@", regex=False)

print("URL STRUCTURE BY CLASS")
print("=" * 60)

for column in [
    "starts_with_https",
    "starts_with_www",
    "has_path",
    "has_query",
    "has_at_symbol"
]:
    print(f"\n{column}")
    print(
        audit.groupby("label")[column]
        .mean()
        .mul(100)
        .rename(index={0: "Phishing", 1: "Legitimate"})
        .round(2)
        .astype(str)
        .add("%")
        .to_string()
    )
    
# STEP 20: COMPLETE THE DATA LEAKAGE AUDIT

import pandas as pd
from urllib.parse import urlsplit

print("=" * 65)
print("STEP 20: DATA LEAKAGE AND COLLECTION BIAS AUDIT")
print("=" * 65)

# Confirm that the expected columns exist
required = ["URL", "label", "URLSimilarityIndex",
            "URLCharProb", "TLDLegitimateProb"]

missing_columns = [col for col in required if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# 1. Check URLSimilarityIndex by class
print("\n1. URLSimilarityIndex == 100 BY CLASS")

print(
    pd.crosstab(
        df["label"].map({0: "Phishing", 1: "Legitimate"}),
        df["URLSimilarityIndex"].eq(100),
        normalize="index"
    ).mul(100).round(2)
)

# 2. Examine URL structure
urls = df["URL"].fillna("").astype(str)

def parse_url(url):
    # Add a scheme only when the URL has none.
    candidate = url if "://" in url else "//" + url
    try:
        return urlsplit(candidate)
    except ValueError:
        return urlsplit("")

parsed = urls.map(parse_url)

url_features = pd.DataFrame({
    "HTTPS": urls.str.lower().str.startswith("https://"),
    "WWW": parsed.map(
        lambda p: p.netloc.lower().startswith("www.")
    ),
    "Path": parsed.map(lambda p: bool(p.path.strip("/"))),
    "Query": parsed.map(lambda p: bool(p.query)),
    "AtSymbol": urls.str.contains("@", regex=False)
}, index=df.index)

print("\n2. URL STRUCTURE BY CLASS (PERCENTAGES)")

for feature in url_features.columns:
    print(f"\n{feature}")
    print(
        pd.crosstab(
            df["label"].map({0: "Phishing", 1: "Legitimate"}),
            url_features[feature],
            normalize="index"
        ).mul(100).round(2)
    )

# 3. Check feature variation within each class
print("\n3. FEATURE VARIATION WITHIN EACH CLASS")

features_to_check = [
    "URLSimilarityIndex",
    "URLCharProb",
    "TLDLegitimateProb",
    "URLLength",
    "DomainLength",
    "IsHTTPS",
    "IsDomainIP"
]

for feature in features_to_check:
    if feature in df.columns:
        print(f"\n{feature}")
        print(
            df.groupby("label")[feature]
              .agg(["count", "nunique", "min", "median", "max"])
              .rename(index={0: "Phishing", 1: "Legitimate"})
        )

# 4. Check whether URLSimilarityIndex == 100 acts
# as a perfect rule for identifying legitimate records
similarity_rule = df["URLSimilarityIndex"].eq(100)
legitimate = df["label"].eq(1)

print("\n4. SIMPLE SIMILARITY RULE CHECK")
print("Records with similarity score 100:", int(similarity_rule.sum()))
print(
    "Legitimate records with score 100:",
    int((similarity_rule & legitimate).sum())
)
print(
    "Phishing records with score 100:",
    int((similarity_rule & df["label"].eq(0)).sum())
)

print("\nAUDIT COMPLETE")

# STEP 21: AUDIT ALL FEATURES FOR POSSIBLE LABEL LEAKAGE

import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 21: FEATURE-LEVEL LEAKAGE AND CLASS-SEPARATION AUDIT")
print("=" * 70)

# Columns that directly identify records or contain raw text.
identifier_or_raw_columns = [
    "FILENAME", "URL", "Domain", "Title", "label"
]

# 1. Find columns with only one value in either class.
print("\n1. FEATURES WITH NO VARIATION WITHIN A CLASS")

for col in df.columns:
    if col == "label":
        continue

    class_unique = df.groupby("label")[col].nunique(dropna=False)

    if (class_unique <= 1).any():
        print(f"\n{col}")
        print("Unique values by class:")
        print(class_unique.rename(index={0: "Phishing", 1: "Legitimate"}))

# 2. Check numerical feature correlations with the label.
print("\n2. NUMERICAL FEATURES MOST CORRELATED WITH THE LABEL")

numeric_df = df.select_dtypes(include="number")
correlations = (
    numeric_df.corr()["label"]
    .drop(labels=["label"], errors="ignore")
    .dropna()
)

print(
    correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    ).head(25).round(4)
)

# 3. Inspect important title/similarity-related features by class.
print("\n3. SELECTED FEATURES BY CLASS")

selected = [
    "URLSimilarityIndex",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "URLCharProb",
    "TLDLegitimateProb",
    "IsHTTPS",
    "IsDomainIP",
    "HasTitle",
    "HasPasswordField",
    "HasExternalFormSubmit"
]

for col in selected:
    if col not in df.columns:
        continue

    print(f"\n{col}")
    print(
        df.groupby("label")[col]
        .agg(["count", "nunique", "min", "median", "max"])
        .rename(index={0: "Phishing", 1: "Legitimate"})
    )

# 4. Check the direct identifiers/raw-text columns.
print("\n4. RAW TEXT AND IDENTIFIER COLUMNS")
for col in identifier_or_raw_columns:
    if col in df.columns:
        print(
            f"{col}: {df[col].nunique(dropna=False)} distinct values"
        )

print("\nSTEP 21 COMPLETE")
