# Common utilities for the matching - used in both EDA & matching code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hipe4ml.tree_handler import TreeHandler
from typing import Optional

DESIGNED_FEATURES = [
    "mchID",
    
    'DeltaX', 'DeltaY', 'DeltaPhi', 'DeltaTanl', 'DeltaR', 'SameSign', 
    
    'PullX', 'PullY', 'PullPhi', 'PullTanl', 'PullR',

    'DeltaDirection',
    
    'PtMCH', 'PtMFT', 'DeltaPt', 'PullPt', 'RelPtDiff',
    ]

NON_TRAINING_FEATURES = [
    'mchID',
    'TimeMCH', 'TimeResMCH', 'TimeMFT', 'TimeResMFT', 
    'MftClusterSizesAndTrackFlags', 
    'Chi2Glob', 'Chi2Match', # what's the diffference???
    'McMaskMCH', 'McMaskMFT', 'McMaskGlob',
    'MatchLabel', 'IsSignal'
    ]

MATCH_LABEL_GROUPS = {
    "Wrong match": [1, 5],
    "Decay":       [2, 6],
    "Fake":        [3, 7],
    "True match":  [0, 4],
}

MATCH_COLOURS = {
    "True match":  "steelblue",
    "Wrong match": "tomato",
    "Decay":       "mediumseagreen",
    "Fake":        "goldenrod",
}


def get_dataframe(file_path: str) -> pd.DataFrame:
    df = TreeHandler(file_path, "O2fwdmlcand", folder_name='DF_*').get_data_frame()
    df.columns = df.columns.str.replace(r'^f', '', regex=True) # Drop leading 'f'
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        print("converting bools to ints for columns:", bool_cols.tolist())
        df[bool_cols] = df[bool_cols].astype(int)
    return df


def design_features(df: pd.DataFrame) -> pd.DataFrame:
    xmch = df['XMCH'].values
    xmft = df['XMFT'].values
    ymch = df['YMCH'].values
    ymft = df['YMFT'].values
    phimch = df['PhiMCH'].values
    phimft = df['PhiMFT'].values
    tanlmch = df['TanlMCH'].values
    tanlmft = df['TanlMFT'].values
    invqptmch = df['InvQPtMCH'].values
    invqptmft = df['InvQPtMFT'].values

    df['DeltaX'] = xmch - xmft
    df['DeltaY'] = ymch - ymft

    dphi = phimch - phimft
    df['DeltaPhi'] = np.arctan2(np.sin(dphi), np.cos(dphi))

    df['DeltaTanl'] = tanlmch - tanlmft

    df['DeltaR'] = np.hypot(df['DeltaX'], df['DeltaY'])
    df['RelPtDiff'] = (1/np.abs(invqptmch) - 1/np.abs(invqptmft))/(1/np.abs(invqptmch)+1/np.abs(invqptmft)) # relative curvature differene

    df['SameSign'] = (np.signbit(invqptmch) == np.signbit(invqptmft)).astype(np.int8)
    df['PtMCH'] = 1 / np.abs(invqptmch) # Rocking only with the MCH Pt for now - gives a consistent value for eventual binning procedure
    df['PtMFT'] = 1 / np.abs(invqptmft)
    df['DeltaPt'] = df['PtMCH'] - df['PtMFT']
    df['PullPt'] = df['DeltaPt'] / np.sqrt(df['C1Pt1PtMCH'] + df['C1Pt1PtMFT']) 

    mch_cols = ["XMCH", "YMCH", "PhiMCH", "TanlMCH", "InvQPtMCH"]
    df["mchID"] = df.round(6).groupby(mch_cols, sort=False).ngroup()

    # check if it's a standard deviation or a variance in the denominator - we will assume it's a variance since that is more common for residual normalization
    df['PullX'] = df['DeltaX'] / np.sqrt(df['CXXMCH'] + df['CXXMFT'])
    df['PullY'] = df['DeltaY'] / np.sqrt(df['CYYMCH'] + df['CYYMFT'])
    df['PullR'] = df['DeltaR'] / np.sqrt(df['CXXMCH'] + df['CXXMFT'] + df['CYYMCH'] + df['CYYMFT'])
    df['PullPhi'] = df['DeltaPhi'] / np.sqrt(df['CPhiPhiMCH'] + df['CPhiPhiMFT'])
    df['PullTanl'] = df['DeltaTanl'] / np.sqrt(df['CTglTglMCH'] + df['CTglTglMFT'])

    df['DeltaDirection'] = np.arccos(
        (np.cos(phimch) * np.cos(phimft) +
         np.sin(phimch) * np.sin(phimft) +
         +tanlmch * tanlmft) / 
        (np.sqrt(1 + tanlmch**2) * np.sqrt(1 + tanlmft**2))
    )
    return df


def perform_cuts(df: pd.DataFrame) -> pd.DataFrame:

    eta_mch = np.arcsinh(pd.to_numeric(df["TanlMCH"], errors="coerce"))
    eta_mask = (eta_mch > -3.6) & (eta_mch < -2.45)

    removed = df[~eta_mask].copy()
    r_rows = int(removed.shape[0])
    r_sig  = int(pd.to_numeric(removed.get("IsSignal", 0), errors="coerce").sum())
    r_bkg  = r_rows - r_sig
    print("[Eta window] -3.6 < eta_MCH < -2.45")
    print(f"Removed rows: {r_rows}  signal={r_sig}  background={r_bkg}")
    df = df[eta_mask].reset_index(drop=True)
    
    # Do the exact same as above but for the PDCA

    # wrap mft phi to [-pi, pi]
    df['PhiMFT'] = np.arctan2(np.sin(df['PhiMFT']), np.cos(df['PhiMFT']))

    return df

def inhousemetrics(df: pd.DataFrame, threshold: float = 0.5, metric: str = "score") -> tuple:
    idx = df.groupby("mchID")[metric].idxmax() # max score index in base df for each mchID group
    best = df.loc[idx].set_index("mchID") # best candidate for each mchID, indexed by mchID
    pairable = df.groupby("mchID")["IsSignal"].any() # Boolean series indicating if each mchID group has at least one true match - indexed by mchID
    total = len(df.groupby("mchID").size())
    

    N_pairable = pairable.sum()

    N_non_pairable = total - N_pairable 

    N_gm_rec = (df.loc[idx, metric] > threshold).sum()

    N_gm_true = (
        (best[metric] > threshold) &
        (best["IsSignal"] == 1)
    ).sum()

    N_gm_rec_pairable = (
        (best[metric] > threshold) &
        pairable
    ).sum()

    N_rejected_non_pairable = (
        (best[metric] <= threshold) &
        (~pairable)
    ).sum()

    N_gm_true_pairable = N_gm_true  # already pairable by construction ---kind of trivializes a bit

    pairing_purity = N_gm_true/N_gm_rec if N_gm_rec > 0 else 0
    pairing_efficiency = N_gm_rec_pairable/N_pairable if N_pairable > 0 else 0
    true_efficiency = N_gm_true_pairable/N_pairable if N_pairable > 0 else 0
    fake_efficiency = (N_gm_rec_pairable - N_gm_true_pairable)/N_pairable if N_pairable > 0 else 0
    rejection_efficiency = N_rejected_non_pairable/N_non_pairable if N_non_pairable > 0 else 0

    return pairing_purity, pairing_efficiency, true_efficiency, fake_efficiency, rejection_efficiency

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics_vs_feature(
    df: pd.DataFrame,
    feature: str,
    threshold: float,
    metrics_fn,
    metric_col_prefix: str = "score",
    n_bins: int = 10,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
):
    """
    Fixed-bin performance vs feature with simple binomial error bars.
    """

    # --- Define bins ---
    # fmin, fmax = df[feature].min(), df[feature].max()

    edges = np.linspace(fmin, fmax, n_bins + 1)

    results = []

    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]

        if i == n_bins - 1:
            df_bin = df[(df[feature] >= low) & (df[feature] <= high)]
        else:
            df_bin = df[(df[feature] >= low) & (df[feature] < high)]

        N = len(df_bin)

        if N == 0:
            continue

        metrics = metrics_fn(df_bin, threshold=threshold, metric=metric_col_prefix)

        # Convert to safe numpy array
        metrics = np.array(metrics, dtype=float)

        # --- crude binomial uncertainty ---
        with np.errstate(invalid='ignore'):
            errors = np.sqrt(metrics * (1 - metrics) / N)

        results.append({
            "bin_low": low,
            "bin_high": high,
            "bin_center": 0.5 * (low + high),
            "bin_width": 0.5 * (high - low),
            "entries": N,
            "pairing_purity": metrics[0],
            "pairing_efficiency": metrics[1],
            "true_efficiency": metrics[2],
            "fake_efficiency": metrics[3],
            "rejection_efficiency": metrics[4],
            "err_pairing_purity": errors[0],
            "err_pairing_efficiency": errors[1],
            "err_true_efficiency": errors[2],
            "err_fake_efficiency": errors[3],
            "err_rejection_efficiency": errors[4],
        })

    result_df = pd.DataFrame(results)

    # --- Plot ---
    plt.figure(figsize=(9, 6))

    metrics_list = [
        "pairing_purity",
        "pairing_efficiency",
        "true_efficiency",
        "fake_efficiency",
        "rejection_efficiency",
    ]

    for col in metrics_list:
        y = result_df[col]
        yerr = result_df[f"err_{col}"]

        # Skip if completely NaN (fixes your missing curve issue)
        if y.isna().all():
            print(f"[WARN] {col} is all NaN → skipped")
            continue

        plt.errorbar(
            result_df["bin_center"],
            y,
            yerr=yerr,
            xerr=result_df["bin_width"],
            fmt='o',
            capsize=3,
            label=col,
        )

    plt.xlabel(feature)
    plt.ylabel("Metric")
    plt.title(f"Metrics vs {feature} (threshold={threshold})")
    plt.legend()
    plt.grid(True)

    plt.show()

    return result_df


def build_match_groups(
    df: pd.DataFrame,
    label_col: str = "MatchLabel",
    label_groups: dict = MATCH_LABEL_GROUPS,
) -> dict:
    """
    Split a dataframe into sub-dataframes by MatchLabel category.
    Returns a dict of {label_name: sub-dataframe}.
    Call once and pass the result to draw_feature().
    """
    return {
        label: df[df[label_col].isin(codes)]
        for label, codes in label_groups.items()
    }


def draw_feature(
    feature: str,
    match_groups: dict,
    colours: dict = MATCH_COLOURS,
    nbins: int = 100,
    per: float = 0.0,
    categorical_max_unique: int = 20,
    density: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a normalised histogram (continuous) or grouped bar chart (categorical)
    of `feature`, broken down by match label category.

    Parameters
    ----------
    feature               : Column name to plot.
    match_groups          : Output of build_match_groups().
    colours               : Dict mapping label name -> matplotlib colour.
    nbins                 : Number of bins for continuous histograms.
    per                   : Quantile to clip outliers at each end (e.g. 0.005).
    categorical_max_unique: Columns with <= this many unique values are treated
                            as categorical and shown as bar charts.
    title                 : Optional plot title. Defaults to the feature name.
    save_path             : If provided, save figure to this path instead of showing.
    """
    # Infer dtype from the first group that has data
    sample_col = next(g[feature] for g in match_groups.values() if len(g) > 0)
    col_dtype  = sample_col.dtype

    is_categorical = (
        col_dtype == bool
        or col_dtype == object
        or pd.api.types.is_integer_dtype(col_dtype)
        and sample_col.nunique() <= categorical_max_unique
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    if is_categorical:
        all_values = sorted(
            set(v for g in match_groups.values() for v in g[feature].unique())
        )
        x     = np.arange(len(all_values))
        width = 0.8 / len(match_groups)

        for i, (label, group) in enumerate(match_groups.items()):
            counts = (
                group[feature]
                .value_counts(normalize=True)
                .reindex(all_values, fill_value=0)
            )
            ax.bar(
                x + i * width,
                counts.values,
                width=width,
                alpha=0.8,
                color=colours.get(label, None),
                label=f"{label}  (n={len(group):,})",
            )

        ax.set_xticks(x + width * (len(match_groups) - 1) / 2)
        ax.set_xticklabels(all_values)
        ax.set_ylabel("Fraction within category", fontsize=20, labelpad=15)

    else:
        minn = max(g[feature].quantile(per)       for g in match_groups.values())
        maxx = min(g[feature].quantile(1 - per)   for g in match_groups.values())

        for label, group in match_groups.items():
            ax.hist(
                group[feature],
                bins=nbins,
                range=(minn, maxx),
                histtype="step",
                linewidth=2,
                alpha=0.8,
                density=density,
                color=colours.get(label, None),
                label=f"{label}  (n={len(group):,})",
            )
    ax.set_ylabel("Normalised to unity" if density else "Counts", fontsize=20, labelpad=15)
    ax.set_xlabel(feature, fontsize=20, labelpad=15)
    ax.set_title(title or feature, fontsize=16)
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(fontsize=13, loc="best", frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def draw_all_features(
    features: list,
    match_groups: dict,
    **kwargs,
) -> None:
    """
    Convenience wrapper to call draw_feature() for a list of features.
    Any keyword arguments are forwarded to draw_feature().
    """
    for feature in features:
        draw_feature(feature, match_groups, **kwargs)