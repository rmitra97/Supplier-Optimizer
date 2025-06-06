# calculate_scores.py
import pandas as pd

def compute_esg_score(csv_path="esg_metrics_combined_predicted.csv"):
    """
    Calculate ESG scores for companies based on their environmental, social, and governance metrics.
    The scoring process involves normalization, confidence weighting, and category weighting.
    
    Args:
        csv_path (str): Path to the CSV file containing ESG metrics
        
    Returns:
        DataFrame: Final ESG scores for each company
    """
    # Load predicted ESG metrics from CSV
    df = pd.read_csv(csv_path)

    # Data cleaning: Remove rows with missing values and convert values to numeric
    df = df.dropna(subset=["value", "unit", "predicted_category", "company"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    # Split metrics into two types for different normalization approaches
    # Percentage metrics (e.g., 50%) and absolute metrics (e.g., tons of CO2)
    percent_df = df[df["unit"].str.contains("%")]
    absolute_df = df[~df["unit"].str.contains("%")]

    def normalize_group(group):
        """
        Normalize values within each group to a 0-1 scale.
        Higher values are better for percentage metrics.
        """
        if group["value"].nunique() == 1:
            group["normalized"] = 1.0  # If all values are same, assign max score
        else:
            # Min-max normalization: (x - min) / (max - min)
            group["normalized"] = (group["value"] - group["value"].min()) / (group["value"].max() - group["value"].min())
        return group

    def normalize_inverse_group(group):
        """
        Inverse normalization for absolute metrics where lower values are better
        (e.g., emissions, waste)
        """
        if group["value"].nunique() == 1:
            group["normalized"] = 1.0
        else:
            # Inverse min-max normalization: 1 - ((x - min) / (max - min))
            group["normalized"] = 1 - ((group["value"] - group["value"].min()) / (group["value"].max() - group["value"].min()))
        return group

    # Apply appropriate normalization to each metric type
    percent_df = percent_df.groupby(["predicted_category", "unit"]).apply(normalize_group)
    absolute_df = absolute_df.groupby(["predicted_category", "unit"]).apply(normalize_inverse_group)

    # Combine normalized metrics back together
    combined_df = pd.concat([percent_df, absolute_df])

    # Apply confidence weighting to account for data reliability
    # Higher confidence metrics have more impact on final score
    confidence_map = {"high": 1.0, "medium": 0.75, "low": 0.5}
    combined_df["confidence_weight"] = combined_df["confidence"].map(confidence_map).fillna(0.5)
    combined_df["weighted_score"] = combined_df["normalized"] * combined_df["confidence_weight"]

    # Define category weights to prioritize different ESG aspects
    # Scope 3 emissions have highest weight as they represent supply chain impact
    category_weights = {
        "Scope 3": 0.35,      # Highest weight for supply chain emissions
        "Scope 1": 0.10,      # Direct emissions
        "Scope 2": 0.10,      # Indirect emissions from purchased energy
        "Waste Reduction": 0.10,
        "Water Reduction": 0.10,
        "Sustainable Packaging": 0.10,
        "Governance": 0.15,   # Second highest weight for governance
        "Other": 0.00        # Unclassified metrics get no weight
    }

    # Apply category weights to get final weighted scores
    combined_df["category_weight"] = combined_df["predicted_category"].map(category_weights).fillna(0)
    combined_df["final_weighted_score"] = combined_df["weighted_score"] * combined_df["category_weight"]

    # Calculate final ESG score per company by summing all weighted scores
    final_scores = combined_df.groupby("company")["final_weighted_score"].sum().reset_index()
    final_scores = final_scores.rename(columns={"final_weighted_score": "final_esg_score"})

    # Save intermediate and final results
    combined_df.to_csv("esg_scores_normalized.csv", index=False)  # Detailed scores with all calculations
    final_scores.to_csv("esg_scores_final.csv", index=False)      # Final company-level scores
    return final_scores
