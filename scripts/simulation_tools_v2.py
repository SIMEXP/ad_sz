import statsmodels.api as sm
import pandas as pd
import numpy as np
import random
import math

from statsmodels.stats.multitest import fdrcorrection
from sklearn.preprocessing import LabelEncoder


def random_sample(df, N):
    # Select N random rows from the DataFrame
    random_indices = random.sample(range(len(df)), N)
    sampled_df = df.iloc[random_indices]

    return sampled_df


def split_sampled_df(sampled_df):
    # Use the sample method with frac=1 to shuffle all rows of the df
    sampled_df_shuffled = sampled_df.sample(frac=1).reset_index(drop=True)

    # Calculate the number of rows to split it in half
    half_rows = len(sampled_df_shuffled) // 2

    # Split the DataFrame into two halves
    group1 = sampled_df_shuffled.iloc[:half_rows]
    group2 = sampled_df_shuffled.iloc[half_rows:]

    return group1, group2


def fisher_transform(df):
    df_transformed = np.arctanh(df)

    return df_transformed


def extract_data(sampled_df, group1, group2):
    # Extract the "site" columns
    group1_site = group1["Site"]
    group2_site = group2["Site"]

    # Convert site values to numeric using label encoding
    le = LabelEncoder()
    le.fit(sampled_df["Site"])
    group1_site = le.transform(group1_site)
    group2_site = le.transform(group2_site)

    # Convert the transformed site values to Pandas Series (for later concatenation)
    group1_site = pd.Series(group1_site)
    group2_site = pd.Series(group2_site)

    # Extract connectome values (excluding "Subject" and "site")
    group1_conn = group1.drop(columns=["Subject", "Site"])
    group2_conn = group2.drop(columns=["Subject", "Site"])

    return group1_site, group2_site, group1_conn, group2_conn


def apply_modification(value, d, std_value):
    return value + d * std_value


def modify_group2(group1_conn, group2_conn, pi, d):
    # Calculate the total number of connections in the DataFrame
    total_conn = group2_conn.shape[1]

    # Calculate the number of connections to modify based on pi%
    num_to_modify = int(total_conn * pi)

    # Randomly select the connections (columns) to modify
    connections_to_modify = group2_conn.sample(n=num_to_modify, axis=1)

    # Stack both groups vertically for std claculation
    combined_data = pd.concat([group1_conn, group2_conn], axis=0)

    # Modify the selected columns in group2
    group2_modified = group2_conn.copy()
    for col in connections_to_modify.columns:
        std = combined_data[col].std()
        group2_modified.loc[:, col] = group2_modified.loc[:, col] + d * std

    return connections_to_modify, group2_modified


def run_cwas(group1_conn, group2_modified, group1_site, group2_site):
    connection_count = group1_conn.shape[1]
    pval_list = []

    for connection_i in range(connection_count):
        # Extract the connectivity data for this connection
        connectivity_i_group1 = group1_conn.iloc[:, connection_i]
        connectivity_i_group2 = group2_modified.iloc[:, connection_i]

        # Stack the connectivity data
        connectivity_data = pd.concat(
            [connectivity_i_group1, connectivity_i_group2], axis=0
        )
        connectivity_data = connectivity_data.astype(float)

        # Create a design matrix with the group (0 or 1) and site information
        design_matrix = pd.DataFrame(
            {
                "Group": np.concatenate(
                    ([0] * len(connectivity_i_group1), [1] * len(connectivity_i_group2))
                ),
                "Site": pd.concat([group1_site, group2_site], axis=0),
                "Constant": 1,
            }
        )
        # Reset index so they match
        connectivity_data.index = design_matrix.index

        # Perform linear regression
        model = sm.OLS(connectivity_data, design_matrix)
        results = model.fit()

        # Save the p values for each connection
        pval = results.pvalues["Group"]
        pval_list.append(pval)

    return pval_list


def run_simulation(conn_df, N, pi, d):
    # Step 1: Randomly select N subjects
    sampled_df = random_sample(conn_df, N)

    # Step 2: Randomly split N selected subjects into 2 groups
    group1, group2 = split_sampled_df(sampled_df)

    group1_site, group2_site, group1_conn, group2_conn = extract_data(
        sampled_df, group1, group2
    )

    # Step 3: Pick pi% of connections at random and modify for group 2
    connections_to_modify, group2_modified = modify_group2(
        group1_conn, group2_conn, pi, d
    )

    # Step 4: Run CWAS
    pval_list = run_cwas(group1_conn, group2_modified, group1_site, group2_site)

    return group1_conn, group2_conn, group2_modified, connections_to_modify, pval_list


def apply_fdr(group1_conn, connections_to_modify, pval_list, q):
    connection_count = group1_conn.shape[1]
    rejected, corrected_pval_list = fdrcorrection(pval_list, alpha=q)

    # Get a list of the modified connections
    modified_conn_list = connections_to_modify.columns.tolist()
    modified_conn_list = [int(conn) for conn in modified_conn_list]

    # Calculate the number of modified connections (condition positive), and non-modified connections (condition negative)
    condition_positive = len(modified_conn_list)
    condition_negative = connection_count - condition_positive
    true_positive_count = 0
    true_negative_count = 0

    for connection in range(connection_count):
        # Connection has been modified, the null hypothesis should be rejected
        if connection in modified_conn_list and rejected[connection]:
            true_positive_count += 1

        # Connection has not been modified, the null hypothesis should not be rejected
        elif connection not in modified_conn_list and not (rejected[connection]):
            true_negative_count += 1

    # Calculate sensitivity and specificity
    sensitivity = true_positive_count / condition_positive
    specificity = true_negative_count / condition_negative

    return corrected_pval_list, sensitivity, specificity


def calculate_theta(N, d):
    # Calculate theta using the formula: theta = d / sqrt(N)
    theta = d / math.sqrt(N)

    return theta


def summary(
    correct_rejected_count, sensitivity_list, specificity_list, d, N, num_sample
):
    # Calculate the estimated statistical power
    power = correct_rejected_count / num_sample

    # Calculate theta (effect size for N = 1)
    theta = calculate_theta(N, d)

    summary_message = (
        f"Estimated power to detect d={d} with N={N}: {power},"
        f" with a mean sensitivity of {round(np.mean(sensitivity_list), 2)} and mean specificity of {round(np.mean(specificity_list), 2)},"
        f" theta (effect size for N=1): {round(theta,2)}"
    )

    return summary_message


def run_simulation_experiment(path_conn, N, pi, d, q, num_sample):
    # Load control connectomes from ABIDE
    conn_df = pd.read_csv(path_conn)

    sensitivity_list = []
    specificity_list = []
    correct_rejected_count = 0
    for sample in range(num_sample):
        # Load connectomes and perform steps 1-4 of simulation
        (
            group1_conn,
            group2_conn,
            group2_modified,
            connections_to_modify,
            pval_list,
        ) = run_simulation(conn_df, N, pi, d)

        # Step 5: Apply FDR correction
        corrected_pval_list, sensitivity, specificity = apply_fdr(
            group1_conn, connections_to_modify, pval_list, q
        )

        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

        # If null hypothesis rejected, plus 1
        if np.any(corrected_pval_list < q):
            correct_rejected_count += 1

    result = summary(
        correct_rejected_count, sensitivity_list, specificity_list, d, N, num_sample
    )

    return group2_conn, group2_modified, connections_to_modify, result
