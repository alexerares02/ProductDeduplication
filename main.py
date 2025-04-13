import pandas as pd
import numpy as np


df = pd.read_csv("veridion_product_deduplication_challenge.csv")

# normalizare
def clean_value(v):
    if pd.isna(v):
        return ""
    return str(v).strip().lower()

# dictionar pentru fiecare rand cu features importante alese manual
def extract_signature_components(row):
    return {
        "unspsc": clean_value(row.get("unspsc", "")),
        "product_name": clean_value(row.get("product_name", "")),
        "intended_industries": clean_value(row.get("intended_industries", "")),
        "applicability": clean_value(row.get("applicability", "")),
        "brand": clean_value(row.get("brand", ""))
    }

df["signature_components"] = df.apply(extract_signature_components, axis=1)

groups = []
assigned = [False] * len(df)

# minim 4 features egale din cele alese manual
def soft_match(comp1, comp2, min_matches=4):
    matches = sum([comp1[k] == comp2[k] for k in comp1])
    return matches >= min_matches

# grupare manuala
for i in range(len(df)):
    if assigned[i]:
        continue
    group = [i]
    for j in range(i + 1, len(df)):
        if not assigned[j] and soft_match(df.at[i, "signature_components"], df.at[j, "signature_components"]):
            group.append(j)
            assigned[j] = True
    assigned[i] = True
    groups.append(group)

# construirea rezultatului
consolidated_rows = []

for group_indices in groups:
    group_df = df.loc[group_indices]
    row = {}
    for col in df.columns:
        if col == "signature_components":
            continue
        if group_df[col].dtype == object:
            vals = group_df[col].dropna().astype(str).unique()
            vals = [v for v in vals if v.strip() and v.strip() != "[]"]
            row[col] = vals[0] if len(vals) == 1 else " || ".join(vals)
        else:
            row[col] = group_df[col].dropna().iloc[0] if not group_df[col].dropna().empty else np.nan
    consolidated_rows.append(row)

# rezultatul final
deduplicated_df = pd.DataFrame(consolidated_rows)

# rezultat final
deduplicated_df.to_csv("veridion_deduplicated_softmatch.csv", index=False)


# raport
initial_count = len(df)
final_count = len(deduplicated_df)
removed_count = initial_count - final_count


print("Raport")
print(f"Initial:{initial_count} randuri")
print(f"După grupare: {final_count} randuri")
