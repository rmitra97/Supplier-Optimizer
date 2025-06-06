
#!pip install pyspark
#!pip install pulp

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, avg
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, lpSum

# Step 1: Start Spark
spark = SparkSession.builder.appName("SupplierOptimizationWithESG").getOrCreate()

# Step 2: Load Supplier Data
unilever_df = spark.read.option("header", True).option("inferSchema", True).csv("Unilever_Supply_Chain.csv")
pg_df = spark.read.option("header", True).option("inferSchema", True).csv("PG_Supply_Chain.csv")
unilever_df = unilever_df.withColumn("Brand", lit("Unilever"))
pg_df = pg_df.withColumn("Brand", lit("Procter & Gamble"))
df = unilever_df.union(pg_df)

# Select needed columns
df = df.select("Product type", "Supplier name", "Brand", "Price", "Lead times", "Defect rates")
# Load ESG Data
esg_df = pd.read_csv("esg_scores_final.csv")
esg_df.columns = esg_df.columns.str.strip()
esg_output = esg_df[["company", "final_esg_score"]]
print("ESG Scores:")
print(esg_output)

# Step 3: Load ESG
esg_df = pd.read_csv("esg_scores_final.csv")
esg_df.columns = esg_df.columns.str.strip()
esg_df.rename(columns={"company": "Brand", "final_esg_score": "ESG_Score"}, inplace=True)

# Normalize ESG
scaler = MinMaxScaler()
esg_df["ESG_Normalized"] = scaler.fit_transform(esg_df[["ESG_Score"]])

# Step 4: Aggregate Supplier Data
brand_avg = df.groupBy("Product type", "Brand").agg(
    avg("Price").alias("avg_price"),
    avg("Lead times").alias("avg_lead_time"),
    avg("Defect rates").alias("avg_defect_rate")
)

# Convert to Pandas
brand_avg_pd = brand_avg.toPandas()
df_pd = df.toPandas()

# Step 5: Merge ESG Scores
brand_avg_pd = brand_avg_pd.merge(esg_df[["Brand", "ESG_Normalized"]], on="Brand", how="left")
mean_esg = esg_df["ESG_Normalized"].mean()
brand_avg_pd["Adjusted_ESG"] = (1 - brand_avg_pd["ESG_Normalized"]).fillna(1 - mean_esg)

# Step 6: Optimization
decision_vars = LpVariable.dicts(
    "Select",
    ((row["Product type"], row["Brand"]) for idx, row in brand_avg_pd.iterrows()),
    cat=LpBinary
)

weights = {
    "Price": 0.3,
    "Lead times": 0.2,
    "Defect rates": 0.3,
    "Adjusted_ESG": 0.2
}

model = LpProblem("Supplier_Selection_Optimization_ESG", LpMinimize)

model += lpSum([
    (weights["Price"] * row["avg_price"] +
     weights["Lead times"] * row["avg_lead_time"] +
     weights["Defect rates"] * row["avg_defect_rate"] +
     weights["Adjusted_ESG"] * row["Adjusted_ESG"]) *
    decision_vars[(row["Product type"], row["Brand"])]
    for idx, row in brand_avg_pd.iterrows()
])

# One Brand per Product Type
for product in brand_avg_pd["Product type"].unique():
    model += lpSum([
        decision_vars[(product, brand)]
        for brand in brand_avg_pd[brand_avg_pd["Product type"] == product]["Brand"].unique()
    ]) == 1

model.solve()

# Step 7: Results
selected_brands = {}
for idx, row in brand_avg_pd.iterrows():
    var_key = (row["Product type"], row["Brand"])
    if decision_vars[var_key].varValue == 1:
        selected_brands[row["Product type"]] = row["Brand"]

final_selection = []
for product, brand in selected_brands.items():
    subset = df_pd[(df_pd["Product type"] == product) & (df_pd["Brand"] == brand)]
    best_row = subset.loc[subset["Price"].idxmin()]
    final_selection.append({
        "Product type": best_row["Product type"],
        "Selected Brand": best_row["Brand"],
        "Best Supplier": best_row["Supplier name"],
        "Price": best_row["Price"],
        "Lead times": best_row["Lead times"],
        "Defect rates": best_row["Defect rates"]
    })

final_selection_df = pd.DataFrame(final_selection)
final_selection_df.to_csv("optimized_supplier_selection_with_esg.csv", index=False)

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

print("\n\nOptimization Complete with ESG!\n")
print(final_selection_df)