import pandas as pd
from pathlib import Path

## Load srpbs participant data and do some stuff

root_p = Path("__file__").resolve().parents[1] / "data"

df = pd.read_csv(root_p / "srpbs" / "participants.tsv", sep="\t")

# Keep only rows where diagnosis is 0 (control) or 4 (sz)
df = df[df["diag"].isin([0, 4])]

# Add columns SZ and CON
df["SZ"] = (df["diag"] == 4).astype(int)
df["CON"] = (df["diag"] == 0).astype(int)

# Add group column
df["group"] = df["diag"]
df["group"] = df["group"].replace({4: "SZ", 0: "CON"})

# Map values for sex, from total N in Tanaka paper
df["sex"] = df["sex"].map({1: 0, 2: 1})

# Add cohort and site columns
df["cohort"] = "SRPBS"

cols = ["participant_id", "age", "sex", "site", "cohort", "group", "SZ", "CON"]
df = df[cols]

# Save results
df.to_csv(root_p / "srpbs_participants.csv", index=False)
