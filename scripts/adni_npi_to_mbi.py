import pandas as pd
from pathlib import Path

## Load NPIQ or NPI scores for ADNI and convert to MBI framework
## Domains: https://adni.bitbucket.io/reference/npi.html


def _mbi_conversion(df):
    # Calculate MBI domains
    df["decreased_motivation"] = df["NPIG"]
    df["emotional_dysregulation"] = df["NPID"] + df["NPIE"] + df["NPIF"]
    df["impulse_dyscontrol"] = df["NPIC"] + df["NPII"] + df["NPIJ"]
    df["social_inappropriateness"] = df["NPIH"]
    df["abnormal_perception"] = df["NPIA"] + df["NPIB"]

    # Calculate MBI total score
    mbi_domains = [
        "decreased_motivation",
        "emotional_dysregulation",
        "impulse_dyscontrol",
        "social_inappropriateness",
        "abnormal_perception",
    ]
    df["mbi_total_score"] = df[mbi_domains].sum(axis=1)
    df["mbi_status"] = (df["mbi_total_score"] >= 1).astype(int)

    mbi_columns = [
        "RID",
        "EXAMDATE",
        "decreased_motivation",
        "emotional_dysregulation",
        "impulse_dyscontrol",
        "social_inappropriateness",
        "abnormal_perception",
        "mbi_total_score",
        "mbi_status",
    ]
    mbi_df = df[mbi_columns].copy()

    return mbi_df


root_p = Path("__file__").resolve().parents[1] / "data" / "adni"

# Load NPI or NPI-Q scores
npi_df = pd.read_csv(root_p / "NPI_22Aug2023.csv")

mbi_df = _mbi_conversion(npi_df)

# Save results - make sure to change name
mbi_df.to_csv(root_p / "adni_npi_mbi_status_20230829.csv", index=False)
