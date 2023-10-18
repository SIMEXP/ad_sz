import pandas as pd
from pathlib import Path

## Load NPIQ scores for CIMA-Q and convert to MBI framework


def _map_values(df):
    # Map values for columns we are using
    mapping = {"0_non": 0, "1_oui_léger": 1, "2_oui_modéré": 2, "3_oui_sévère": 3}

    columns_to_map = [
        "22901_apathie",
        "22901_depression_dysphorie",
        "22901_anxiete",
        "22901_euphorie",
        "22901_agitation_aggressivite",
        "22901_irritabilite",
        "22901_comp_moteur_aberrant",
        "22901_impulsivite",
        "22901_idees_delirantes",
        "22901_hallucinations",
    ]

    for column in columns_to_map:
        df[column] = df[column].map(mapping)

    return df


def _mbi_conversion(df):
    # Calculate MBI domains
    df["decreased_motivation"] = df["22901_apathie"]
    df["emotional_dysregulation"] = (
        df["22901_depression_dysphorie"] + df["22901_anxiete"] + df["22901_euphorie"]
    )
    df["impulse_dyscontrol"] = (
        df["22901_agitation_aggressivite"]
        + df["22901_irritabilite"]
        + df["22901_comp_moteur_aberrant"]
    )
    df["social_inappropriateness"] = df["22901_impulsivite"]
    df["abnormal_perception"] = (
        df["22901_idees_delirantes"] + df["22901_hallucinations"]
    )

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
        "PSCID",
        "Date_taken",
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


root_p = Path("__file__").resolve().parents[1] / "data" / "cimaq"

# Load NPI-Q scores
npi_df = pd.read_csv(
    root_p / "22901_inventaire_neuropsychiatrique_q_initiale.tsv", sep="\t"
)

npi_df = npi_df.dropna(subset=["Date_taken"])
npi_df = _map_values(npi_df)
mbi_df = _mbi_conversion(npi_df)

# Save results - make sure to change name
mbi_df.to_csv(root_p / "cimaq_npiq_mbi_status_20230829.csv", index=False)
