import pandas as pd
from pathlib import Path

## Load cobre participant data and do some stuff

root_p = Path("__file__").resolve().parents[1] / "data"

df = pd.read_csv(root_p / "cobre" / "COBRE_phenotypic_data.csv")

# Add columns SZ and CON
df["SZ"] = (df["Subject Type"] == "Patient").astype(int)
df["CON"] = (df["Subject Type"] == "Control").astype(int)

# Add group column
df["group"] = df["Subject Type"]
df["group"] = df["group"].replace({"Patient": "SZ", "Control": "CON"})

# Rename some columns and map values for sex
df.rename(
    columns={"Gender": "sex", "Current Age": "age", "Unnamed: 0": "participant_id"},
    inplace=True,
)
df["sex"] = df["sex"].map({"Male": 0, "Female": 1})

# Add cohort and site columns
df["cohort"] = "COBRE"
df["site"] = 0

cols = ["participant_id", "age", "sex", "site", "cohort", "group", "SZ", "CON"]
df = df[cols]

# Save results
df.to_csv(root_p / "cobre_participants.csv", index=False)
