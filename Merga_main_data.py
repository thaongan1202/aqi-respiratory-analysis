import pandas as pd
import re
import unicodedata
from pathlib import Path

# ============================================================
# 1. KHAI BAO FILE
# PLACES release 2021 -> du lieu suc khoe chu yeu nam 2019
# PLACES release 2022 -> 2020
# PLACES release 2023 -> 2021
# PLACES release 2024 -> 2022
# ============================================================

base_path = r"D:\TT Dữ liệu trực quan\ProjectCuoiKy\Dataset\AQI — EPA AirData"

aqi_files = {
    2019: base_path + r"\annual_aqi_by_county_2019.csv",
    2020: base_path + r"\annual_aqi_by_county_2020.csv",
    2021: base_path + r"\annual_aqi_by_county_2021.csv",
    2022: base_path + r"\annual_aqi_by_county_2022.csv"
}

places_files = {
    2019: base_path + r"\PLACES_County_Data_(GIS_Friendly_Format),_2021_release.csv",
    2020: base_path + r"\PLACES_County_Data_(GIS_Friendly_Format),_2022_release.csv",
    2021: base_path + r"\PLACES_County_Data_(GIS_Friendly_Format),_2023_release.csv",
    2022: base_path + r"\PLACES_County_Data_(GIS_Friendly_Format),_2024_release.csv"
}


# ============================================================
# 2. HAM CHUAN HOA TEN BANG / COUNTY
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return ""

    x = unicodedata.normalize("NFKD", str(x))
    x = x.encode("ascii", "ignore").decode()
    x = x.upper().strip()

    x = re.sub(
        r"\b(COUNTY|PARISH|BOROUGH|CENSUS AREA|MUNICIPALITY|CITY AND BOROUGH)\b",
        "",
        x
    )

    x = re.sub(r"[^A-Z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x


# ============================================================
# 3. CAC BIEN CAN GIU
# ============================================================

aqi_keep = [
    "State",
    "County",
    "Year",
    "Days with AQI",
    "Good Days",
    "Moderate Days",
    "Unhealthy for Sensitive Groups Days",
    "Unhealthy Days",
    "Very Unhealthy Days",
    "Hazardous Days",
    "Max AQI",
    "90th Percentile AQI",
    "Median AQI",
    "Days NO2",
    "Days Ozone",
    "Days PM2.5",
    "Days PM10"
]

health_keep = [
    "StateAbbr",
    "StateDesc",
    "CountyName",
    "CountyFIPS",
    "TotalPopulation",
    "CASTHMA_AdjPrev",
    "COPD_AdjPrev",
    "CSMOKING_AdjPrev",
    "Geolocation"
]


# ============================================================
# 4. DOC + MERGE TUNG NAM
# ============================================================

merged_years = []
merge_statistics = []

for year in range(2019, 2023):

    print("\n============================")
    print("YEAR:", year)
    print("============================")

    # Doc EPA AQI
    aqi = pd.read_csv(
        aqi_files[year],
        usecols=aqi_keep
    )

    # Doc CDC PLACES
    places = pd.read_csv(
        places_files[year],
        usecols=health_keep,
        dtype={"CountyFIPS": str}
    )

    places["CountyFIPS"] = places["CountyFIPS"].str.zfill(5)

    # Tao khoa merge tam thoi
    aqi["_state"] = aqi["State"].apply(normalize_text)
    aqi["_county"] = aqi["County"].apply(normalize_text)

    places["_state"] = places["StateDesc"].apply(normalize_text)
    places["_county"] = places["CountyName"].apply(normalize_text)

    # --------------------------------------------------------
    # Xu ly cac ten county/city bi trung sau khi chuan hoa.
    # Loai cac khoa mo ho de tranh merge nham.
    # --------------------------------------------------------

    duplicate_keys = (
        places.loc[
            places.duplicated(
                subset=["_state", "_county"],
                keep=False
            ),
            ["_state", "_county"]
        ]
        .drop_duplicates()
    )

    if len(duplicate_keys) > 0:

        duplicate_index = pd.MultiIndex.from_frame(
            duplicate_keys
        )

        places_index = pd.MultiIndex.from_frame(
            places[["_state", "_county"]]
        )

        places = places[
            ~places_index.isin(duplicate_index)
        ].copy()

    # Merge
    merged = pd.merge(
        aqi,
        places,
        on=["_state", "_county"],
        how="inner",
        validate="many_to_one"
    )

    print("EPA rows:", len(aqi))
    print("Matched rows:", len(merged))
    print(
        "Match rate:",
        round(len(merged) / len(aqi) * 100, 2),
        "%"
    )

    merge_statistics.append({
        "Year": year,
        "EPA_rows": len(aqi),
        "Matched_rows": len(merged),
        "Match_rate": round(
            len(merged) / len(aqi) * 100,
            2
        ),
        "Asthma_missing":
            merged["CASTHMA_AdjPrev"].isna().sum(),
        "COPD_missing":
            merged["COPD_AdjPrev"].isna().sum(),
        "Smoking_missing":
            merged["CSMOKING_AdjPrev"].isna().sum()
    })

    merged_years.append(merged)


# ============================================================
# 5. GOM 4 NAM
# ============================================================

df = pd.concat(
    merged_years,
    ignore_index=True
)

print("\nRows after merge:", len(df))


# ============================================================
# 6. GIU CAC BIEN PHU HOP VOI DE TAI
# ============================================================

df = df[[
    "CountyFIPS",
    "CountyName",
    "StateAbbr",
    "StateDesc",
    "Year",
    "TotalPopulation",
    "Geolocation",

    # AQI
    "Days with AQI",
    "Good Days",
    "Moderate Days",
    "Unhealthy for Sensitive Groups Days",
    "Unhealthy Days",
    "Very Unhealthy Days",
    "Hazardous Days",
    "Max AQI",
    "90th Percentile AQI",
    "Median AQI",

    # Chat o nhiem quyet dinh AQI
    "Days NO2",
    "Days Ozone",
    "Days PM2.5",
    "Days PM10",

    # Benh ho hap + bien kiem soat
    "CASTHMA_AdjPrev",
    "COPD_AdjPrev",
    "CSMOKING_AdjPrev"
]].copy()


# ============================================================
# 7. DOI TEN COT CHO DE SU DUNG
# ============================================================

df.columns = [
    "CountyFIPS",
    "County",
    "StateAbbr",
    "State",
    "Year",
    "Population",
    "Geolocation",

    "Days_with_AQI",
    "Good_Days",
    "Moderate_Days",
    "USG_Days",
    "Unhealthy_Days",
    "Very_Unhealthy_Days",
    "Hazardous_Days",

    "Max_AQI",
    "AQI_90th_Percentile",
    "Median_AQI",

    "Days_NO2",
    "Days_Ozone",
    "Days_PM2_5",
    "Days_PM10",

    "Asthma_AdjPrev",
    "COPD_AdjPrev",
    "Smoking_AdjPrev"
]


# ============================================================
# 8. LUU BAN WIDE
# Moi County-Year = 1 dong
# Asthma va COPD nam o 2 cot rieng
# ============================================================

df.to_csv(
    "AQI_Respiratory_Disease_2019_2022_Wide.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. CHUYEN SANG LONG FORMAT
#
# Don vi phan tich:
# County x Year x Disease
#
# Vi du:
# Los Angeles | 2019 | Current Asthma | 9.1
# Los Angeles | 2019 | COPD           | 5.2
# ============================================================

id_columns = [
    col for col in df.columns
    if col not in [
        "Asthma_AdjPrev",
        "COPD_AdjPrev"
    ]
]

df_long = df.melt(
    id_vars=id_columns,
    value_vars=[
        "Asthma_AdjPrev",
        "COPD_AdjPrev"
    ],
    var_name="Disease",
    value_name="Disease_AdjPrev"
)

df_long["Disease"] = df_long["Disease"].replace({
    "Asthma_AdjPrev": "Current Asthma",
    "COPD_AdjPrev": "COPD"
})


# ============================================================
# 10. XU LY MISSING
#
# Chi drop neu thieu bien THIET YEU cho phan tich.
# Khong drop tat ca cac cot mot cach may moc.
# ============================================================

essential_columns = [
    "CountyFIPS",
    "Year",
    "Disease",
    "Disease_AdjPrev",
    "Median_AQI",
    "Max_AQI",
    "Days_with_AQI"
]

df_clean = df_long.dropna(
    subset=essential_columns
).copy()


# ============================================================
# 11. SAP XEP
# ============================================================

df_clean = df_clean.sort_values(
    by=[
        "Year",
        "State",
        "County",
        "Disease"
    ]
).reset_index(drop=True)


# ============================================================
# 12. KIEM TRA DUPLICATE
# ============================================================

duplicate_count = df_clean.duplicated(
    subset=[
        "CountyFIPS",
        "Year",
        "Disease"
    ]
).sum()

print(
    "\nDuplicate County-Year-Disease:",
    duplicate_count
)


# ============================================================
# 13. KIEM TRA KET QUA
# ============================================================

print("\n================================")
print("FINAL DATASET")
print("================================")

print("Shape:", df_clean.shape)
print("Rows:", len(df_clean))
print("Columns:", len(df_clean.columns))

print(
    "Unique counties:",
    df_clean["CountyFIPS"].nunique()
)

print(
    "Years:",
    sorted(df_clean["Year"].unique())
)

print("\nRows by disease:")
print(
    df_clean["Disease"].value_counts()
)

print("\nMissing values:")
missing = df_clean.isna().sum()
print(
    missing[missing > 0]
)

print("\nMerge statistics:")
print(
    pd.DataFrame(merge_statistics)
)


# ============================================================
# 14. LUU DATASET CUOI
# ============================================================

base_path = Path(__file__).resolve().parent

output_file = base_path / "AQI_Respiratory_Disease_2019_2022_Merged_Clean.csv"

df_clean.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nĐã lưu file tại:")
print(output_file)