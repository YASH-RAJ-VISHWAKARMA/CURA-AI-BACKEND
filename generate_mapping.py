# generate_mapping.py
import pandas as pd
from utils_data import load_and_clean, generate_doctor_mapping
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(BASE_DIR, "disease_doctor_mapping.csv")

train_df, _ = load_and_clean()
diseases = train_df['prognosis'].unique()
mapping = generate_doctor_mapping(diseases)

# Save to CSV so you can manually refine
rows = [[d, mapping[d]] for d in mapping]
pd.DataFrame(rows).to_csv(OUT_CSV, index=False, header=False)
print("Saved seed mapping to:", OUT_CSV)
