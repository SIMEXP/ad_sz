import pandas as pd
from pathlib import Path

## Load ds000030 participant data and do some stuff

root_p = Path("__file__").resolve().parents[1] / "data"

df = pd.read_csv(root_p / "ds000030" / "participants.csv")

# Keep only rows where diagnosis is CONTROL or SCHZ
df = df[df["diagnosis"].isin(["CONTROL", "SCHZ"])]

# Add columns SZ and CON
df["SZ"] = (df["diagnosis"] == "SCHZ").astype(int)
df["CON"] = (df["diagnosis"] == "CONTROL").astype(int)

# Add group column
df["group"] = df["diagnosis"]
df["group"] = df["group"].replace({"SCHZ": "SZ", "CONTROL": "CON"})

# Rename some columns and map values for sex
df.rename(columns={"gender": "sex"}, inplace=True)
df["sex"] = df["sex"].map({"M": 0, "F": 1})

# Add COHORT and SITE columns
df["cohort"] = "ds000030"
df["site"] = 0

cols = ["participant_id", "age", "sex", "site", "cohort", "group", "SZ", "CON"]
df = df[cols]

# Save results
df.to_csv(root_p / "ds000030_participants.csv", index=False)
