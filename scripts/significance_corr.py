import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import util
import time
import os
import itertools
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr
from argparse import ArgumentParser


# SEBASTIEN URCHS
def p_permut(empirical_value, permutation_values):
    n_permutation = len(permutation_values)
    if empirical_value >= 0:
        return (np.sum(permutation_values > empirical_value) + 1) / (n_permutation + 1)
    return (np.sum(permutation_values < empirical_value) + 1) / (n_permutation + 1)


def filter_fdr(df, contrasts):
    df_filtered = df[
        (df["pair0"].isin(contrasts)) & (df["pair1"].isin(contrasts))
    ].copy()
    _, fdr, _, _ = multipletests(df_filtered["pval"], method="fdr_bh")
    df_filtered["fdr_filtered"] = fdr
    return df_filtered


def mat_form(df, contrasts, value="betamap_corr"):
    n = len(contrasts)
    d = dict(zip(contrasts, range(n)))
    mat = np.zeros((n, n))
    for c in contrasts:
        # fill out vertical strip of mat
        for i in range(n):
            if i == d[c]:
                val = 1
            else:
                val = df[
                    ((df["pair0"] == c) | (df["pair1"] == c))
                    & ((df["pair0"] == contrasts[i]) | (df["pair1"] == contrasts[i]))
                ][value]
            mat[i, d[c]] = val
            mat[d[c], i] = val
    return pd.DataFrame(mat, columns=contrasts, index=contrasts)


def make_matrices(df, contrasts, fdr="fdr_filtered"):
    "Param fdr can be set to 'fdr_filtered': FDR is performed using the pvalues only from the chosen contrasts"
    "                              or 'fdr': values taken from FDR performed on full set of 42 contrasts"
    if fdr == "fdr_filtered":
        df = filter_fdr(df, contrasts)
    mat_corr = mat_form(df, contrasts, value="betamap_corr")
    mat_pval = mat_form(df, contrasts, value="pval")
    mat_fdr = mat_form(df, contrasts, value=fdr)
    return mat_corr, mat_pval, mat_fdr


def get_corr_dist(cases, nulls, path_out, tag="wholeconn"):
    # For each unique pair, between the null maps.
    n_pairs = int((len(cases)) * (len(cases) - 1) / 2)
    corr = np.zeros((n_pairs, 5000))

    print(
        "Getting correlation between 5000 null maps for {} unique pairs for {} cases...".format(
            n_pairs, len(cases)
        )
    )
    pair = []
    l = 0
    for i in itertools.combinations(cases, 2):
        for j in range(5000):
            corr[l, j] = pearsonr(
                nulls.loc[i[0]].values[j, :], nulls.loc[i[1]].values[j, :]
            )[0]

        pair.append(i)
        if l % 50 == 0:
            print("{}/{}".format(l, n_pairs))
        l = l + 1

    df = pd.DataFrame(corr)
    df["pair"] = pair
    df.to_csv(os.path.join(path_out, "correlation_dist_{}.csv".format(tag)))
    return df


def get_corr(cases, betas, path_out, tag="wholeconn"):
    # For each unique pair, correlation between betamaps. Use standardized betas here (as in rest of paper).
    n_pairs = int((len(cases)) * (len(cases) - 1) / 2)
    corr = np.zeros(n_pairs)

    print(
        "Getting correlation between betamaps for {} unique pairs for {} cases...".format(
            n_pairs, len(cases)
        )
    )
    pair = []
    l = 0
    for i in itertools.combinations(cases, 2):
        corr[l] = pearsonr(betas.loc[i[0]].values, betas.loc[i[1]].values)[0]
        l = l + 1
        pair.append(i)
    df = pd.DataFrame(corr)
    df["pair"] = pair
    df.to_csv(os.path.join(path_out, "correlation_betas_{}.csv".format(tag)))
    return df


def get_corr_pval(maps, nulls, betas, path_out, tag="wholeconn"):
    df = get_corr_dist(maps, nulls, path_out, tag=tag)
    df_bb = get_corr(maps, betas, path_out, tag=tag)

    df_bb = df_bb.rename(columns={0: "betamap_corr"})
    df_master = df_bb.merge(df, on="pair")

    print("Calculating pvals...")
    # CALCULATE PVALS
    pval = []
    for i in df_master.index:
        p = p_permut(df_master.loc[i, "betamap_corr"], df_master[range(5000)].loc[i])
        pval.append(p)
    df_master["pval"] = pval

    # ADD LABELS
    pair0 = [p[0] for p in df["pair"].tolist()]
    pair1 = [p[1] for p in df["pair"].tolist()]
    df_master["pair0"] = pair0
    df_master["pair1"] = pair1

    df_compact = df_master[["pair0", "pair1", "betamap_corr", "pval"]]
    df_compact.to_csv(
        os.path.join(path_out, "corr_pval_null_v_null_{}.csv".format(tag))
    )

    return df_compact


def _make_region_masks(row):
    mask = np.tri(64, k=0, dtype=bool)

    # Get the parcel number from the beginning of the row and subtract 1
    parcel_num = int(row[0]) - 1

    # Get the parcel name from the second column value
    parcel_name = str(row[1]).upper().replace(" ", "_")

    # Generate the mask
    parcel = np.zeros((64, 64), bool)
    parcel[:, parcel_num] = True
    parcel_mask = parcel + np.transpose(parcel)
    parcel_mask = np.tril(parcel_mask)
    parcel_mask = parcel_mask[mask]

    return (parcel_mask, parcel_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--n_path_mc", help="path to mc null models dir", dest="n_path_mc"
    )
    parser.add_argument("--b_path_mc", help="path to mc betamaps dir", dest="b_path_mc")
    parser.add_argument("--path_out", help="path to output directory", dest="path_out")
    parser.add_argument(
        "--path_corr", help="path to corr dir", dest="path_corr", default=None
    )
    parser.add_argument(
        "--path_parcellation", help="path to MIST_64.cv", dest="path_parcellation"
    )
    args = parser.parse_args()

    n_path_mc = os.path.join(args.n_path_mc, "null_model_{}_vs_control_mc.npy")
    # cont_n_path_mc = os.path.join(args.n_path_mc,'{}_null_model_mc.npy')
    b_path_mc = os.path.join(args.b_path_mc, "{}_vs_con_mean.csv")
    # cont_b_path_mc = os.path.join(args.b_path_mc,'cont_{}_results_mc.csv')
    path_out = args.path_out
    path_corr = args.path_corr

    cases = ["mci_neg", "mci_pos", "sz"]

    maps = cases  # + cont

    #############
    # LOAD DATA #
    #############
    null = []
    beta_std = []

    for c in cases:
        null.append(pd.DataFrame(np.load(n_path_mc.format(c))))
        beta_std.append(pd.read_csv(b_path_mc.format(c))["stand_betas"].values)

    betamaps_std = pd.DataFrame(beta_std, index=maps)
    nullmodels = pd.concat(null, keys=maps)

    ####################
    # WHOLE CONNECTOME #
    ####################
    print("Creating correlation distributions for whole connectome...")
    df_wc = get_corr_pval(maps, nullmodels, betamaps_std, path_out, tag="wholeconn")

    # make matrices
    print("Preparing correlation matrices...")
    subset = ["mci_neg", "mci_pos", "sz"]

    corr, pval, fdr = make_matrices(df_wc, subset, fdr="fdr_filtered")

    corr.to_csv(os.path.join(path_out, "FC_corr_wholebrain_mc_null_v_null.csv"))
    pval.to_csv(os.path.join(path_out, "FC_corr_pval_wholebrain_mc_null_v_null.csv"))
    fdr.to_csv(
        os.path.join(path_out, "FC_corr_fdr_filtered_wholebrain_mc_null_v_null.csv")
    )

    ####################
    # REGIONS #
    ####################
    df_MIST = pd.read_csv(args.path_parcellation, delimiter=";")

    for index, row in df_MIST.iterrows():
        # create region mask
        parcel_mask, parcel_name = _make_region_masks(row)

        # filter maps
        null_parcel = [n.transpose()[parcel_mask].transpose() for n in null]
        beta_std_parcel = [b[parcel_mask] for b in beta_std]

        betamaps_std_parcel = pd.DataFrame(beta_std_parcel, index=maps)
        nullmodels_parcel = pd.concat(null_parcel, keys=maps)

        print(f"Creating correlation distributions for {parcel_name}...")
        df_parcel = get_corr_pval(
            maps, nullmodels_parcel, betamaps_std_parcel, path_out, tag=parcel_name
        )

        # make matrices
        corr_parcel, pval_parcel, fdr_filtered_parcel = make_matrices(
            df_parcel, subset, fdr="fdr_filtered"
        )

        _, pval_parcel_fdr_corrected, _, _ = multipletests(pval_parcel, method="fdr_bh")

        # TO DO - see if I need to correct fdr filtered, I think so?
        if (pval_parcel < 0.05).any():
            corr_parcel.to_csv(
                os.path.join(path_out, f"FC_corr_{parcel_name}_mc_null_v_null.csv")
            )
            pval_parcel.to_csv(
                os.path.join(path_out, f"FC_corr_pval_{parcel_name}_mc_null_v_null.csv")
            )
            pval_parcel_fdr_corrected.to_csv(
                os.path.join(
                    path_out,
                    f"FC_corr_pval_fdr_corrected_{parcel_name}_mc_null_v_null.csv",
                )
            )
            fdr_filtered_parcel.to_csv(
                os.path.join(
                    path_out, f"FC_corr_fdr_filtered_{parcel_name}_mc_null_v_null.csv"
                )
            )

    print("Done!")
