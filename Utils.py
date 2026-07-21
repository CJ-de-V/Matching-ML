# Common utilities for the matching - used in both EDA & matching code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hipe4ml.tree_handler import TreeHandler
from typing import Optional, Union


#TODO: Consider adding absolute value features for the symmetric(ish) pulls/deltas, add DCA_XY ALSO add some metric of total uncertainty... like something derived of the whole covariance matrix
DESIGNED_FEATURES = [
    "mchID", "is_dummy", # Indexing and dummy flag
    
    'DeltaX', 'DeltaY', 'DeltaPhi', 'DeltaTanl', 'DeltaR', # MFT-MCH feature residuals
    
    'SameSign', # SignMch == SignMFT
    
    'DCAXY', 'RMFT'
    
    'PullX', 'PullY', 'PullPhi', 'PullTanl', 'PullR', # residuals / sqrt(Cfeaturefeature) from covariance matrix

    'DeltaDirection', # angle between MCH and MFT track directions
    
    'PtMCH', 'PtMFT', 'DeltaPt', 'PullPt', 'RelPtDiff', # Pt difference / sum of Pt magnitudes

    'etaMCH', 'etaMFT', 'DeltaEta', 

    'ADeltaX', 'ADeltaY', 'ADeltaPhi', # Absolute value of the deltas - maybe not useful but could be for the model to have access to the magnitude of the disagreement regardless of direction
    
    'APullX', 'APullY', 'APullPhi',# Absolute value of the pulls - maybe not useful but could be for the model to have access to the magnitude of the disagreement regardless of direction
    ]

NON_TRAINING_FEATURES = [
    'mchID',
    'TimeMCH', 'TimeResMCH', 'TimeMFT', 'TimeResMFT', 
    'MftClusterSizesAndTrackFlags', 
    'Chi2Glob', 'Chi2Match', # temporarily included in training and evaluation
    'McMaskMCH', 'McMaskMFT', 'McMaskGlob',
    'MatchLabel', 'IsSignal',
    #, 'is_dummy', # Added is_dummy as we are not currently using it and don't want it in the o2 without use

    # features we exclude based on bias & selection cuts, i.e. we do not want our model to discriminate based on these, since it learns what we tell it to, not what we intend for it to
    # Otherwise we risk it learning for example that non-prompts are bad
    'DCAX', 'DCAY', 'PDCA', 'DCAXY', 'Rabs', 'IsAmbig', 'Chi2MCH', 'Chi2MFT', 'MFTMult'
    ]

# TODO: ensure these features are not read in training to save on space
SKIPPED_FEATURES  = [
    'TimeMCH', 'TimeResMCH', 'TimeMFT', 'TimeResMFT', 
    'MftClusterSizesAndTrackFlags', 
    'Chi2Glob', 'Chi2Match', # temporarily included in training and evaluation
    'McMaskMCH', 'McMaskMFT', 'McMaskGlob',
    ]

#TODO: use this instead of the mch features to assign groups, do metric plotting as a function of these for evluation
GROUP_PRESERVING_FEATURES = [
    'XMCH', 'YMCH', 'PhiMCH', 'TanlMCH', 'InvQPtMCH', "chi2MCH",
    'Chi2MCH', 'PDCA', 'Rabs', 'CXXMCH', 'CYYMCH', 'CPhiPhiMCH', 'CTglTglMCH', 'C1Pt1PtMCH'
]

MATCH_LABEL_GROUPS = {
    "True":  [0, 4],
    "Wrong": [1, 5],
    "Decay": [2, 6],
    "Fake":  [3, 7],
    "Unknown": [8], 
}

MATCH_COLOURS = {
    "True":  "black",
    "Wrong": "red",
    "Decay": "lime",
    "Fake":  "mediumblue",
    "Unknown": "gray",
}

def subsample(df: pd.DataFrame, frac: float = 0.5) -> pd.DataFrame:
    """Downsample the df while preserving group structure"""
    unique_ids = df["mchID"].unique()
    n_sample = int(frac * len(unique_ids))
    sampled_ids = np.random.choice(unique_ids, n_sample, replace=False)
    df = df[df["mchID"].isin(sampled_ids)]
    return df


def get_dataframe(file_path: str, folder_name: str ) -> pd.DataFrame:
    df = TreeHandler(file_path, "O2fwdmlcand", folder_name=folder_name).get_data_frame()
    df.columns = df.columns.str.replace(r'^f', '', regex=True) # Drop leading 'f'
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        print("converting bools to ints for columns:", bool_cols.tolist())
        df[bool_cols] = df[bool_cols].astype(int)
    return df

def process_dataframe(df: pd.DataFrame, makedummies: bool) -> pd.DataFrame:
    # --- 1. Perform cuts ---
    df = perform_cuts(df) 
    print(f"After cuts, shape: {df.shape}")
    # --- 2. Design features ---
    df = design_features(df)
    print(f"After feature design, shape: {df.shape}")
    # --- 3. Add dummy candidates for non-pairable groups ---
    if makedummies:
        df = add_dummy_candidates(df, FEATURES=[f for f in df.columns.tolist() if f not in NON_TRAINING_FEATURES], group_col="mchID", signal_col="IsSignal", matchlabel_col="MatchLabel", dummy_flag_col="is_dummy")
        print(f"Added dummy candidates. New shape: {df.shape}")
    
    return df


def design_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df['etaMCH'] = np.arcsinh(df['TanlMCH']).astype(np.float32)
    df['etaMFT'] = np.arcsinh(df['TanlMFT']).astype(np.float32)
    df['DeltaEta'] = (df['etaMCH'] - df['etaMFT']).astype(np.float32)

    df['DCAXY'] = np.sqrt(df['DCAX']**2 + df['DCAY']**2).astype(np.float32)

    df["is_dummy"] = 0 # ensure the column exists even if we are not adding dummy candidates - will be 0 for all real candidates

    df['DeltaX'] = (df['XMCH'] - df['XMFT']).astype(np.float32)
    df['DeltaY'] = (df['YMCH'] - df['YMFT']).astype(np.float32)
    dphi = (df['PhiMCH'] - df['PhiMFT']).astype(np.float32)
    df['DeltaPhi'] = np.arctan2(np.sin(dphi), np.cos(dphi)).astype(np.float32)
    df['ADeltaPhi'] = np.abs(df['DeltaPhi']).astype(np.float32)
    df['ADeltaX'] = np.abs(df['DeltaX']).astype(np.float32)
    df['ADeltaY'] = np.abs(df['DeltaY']).astype(np.float32)

    df['DeltaTanl'] = (df['TanlMCH'] - df['TanlMFT']).astype(np.float32)

    df['DeltaR'] = np.hypot(df['DeltaX'], df['DeltaY']).astype(np.float32)
    df['RMFT'] = np.hypot(df['XMFT'], df['YMFT']).astype(np.float32)

    df['SameSign'] = (np.signbit(df['InvQPtMCH']) == np.signbit(df['InvQPtMFT'])).astype(np.int8)
    df['PtMCH'] = (1 / np.abs(df['InvQPtMCH'])).astype(np.float32) # Rocking only with the MCH Pt for now - gives a consistent value for eventual binning procedure
    df['PtMFT'] = (1 / np.abs(df['InvQPtMFT'])).astype(np.float32)
    df['DeltaPt'] = (df['PtMCH'] - df['PtMFT']).astype(np.float32)
    df['RelPtDiff'] = (df['DeltaPt'] / (df['PtMFT'] + df['PtMCH'])).astype(np.float32) # relative curvature difference
    df['PullPt'] = (df['DeltaPt'] / np.sqrt(df['C1Pt1PtMCH']/df['InvQPtMCH']**4 + df['C1Pt1PtMFT']/df['InvQPtMFT']**4)).astype(np.float32) # error stored is 1/pt's TODO: Fix to properly use uncertainties on Pt instead of 1/Pt

    mch_cols = ["XMCH", "YMCH", "PhiMCH", "TanlMCH", "InvQPtMCH"]
    group_keys = df[mch_cols].round(6)
    df["mchID"] = (
        group_keys.groupby(mch_cols, sort=False, dropna=False)
        .ngroup()
        .astype(np.int32, copy=False)
    )

    df['PullX'] = (df['DeltaX'] / np.sqrt(df['CXXMCH'] + df['CXXMFT'])).astype(np.float32)
    df['PullY'] = (df['DeltaY'] / np.sqrt(df['CYYMCH'] + df['CYYMFT'])).astype(np.float32)
    df['PullR'] = (df['DeltaR'] / np.sqrt(df['CXXMCH'] + df['CXXMFT'] + df['CYYMCH'] + df['CYYMFT'])).astype(np.float32)
    df['PullPhi'] = (df['DeltaPhi'] / np.sqrt(df['CPhiPhiMCH'] + df['CPhiPhiMFT'])).astype(np.float32)
    df['PullTanl'] = (df['DeltaTanl'] / np.sqrt(df['CTglTglMCH'] + df['CTglTglMFT'])).astype(np.float32)
    df['APullX'] = np.abs(df['PullX']).astype(np.float32)
    df['APullY'] = np.abs(df['PullY']).astype(np.float32)
    df['APullPhi'] = np.abs(df['PullPhi']).astype(np.float32)



    cos_delta = (np.cos(df['PhiMCH']) * np.cos(df['PhiMFT']) + np.sin(df['PhiMCH']) * np.sin(df['PhiMFT']) + df['TanlMCH'] * df['TanlMFT']) / (np.sqrt(1 + df['TanlMCH']**2) * np.sqrt(1 + df['TanlMFT']**2))
    df['DeltaDirection'] = np.arccos(np.clip(cos_delta, -1, 1)).astype(np.float32) # Clip for numerical stability
    return df

def add_dummy_candidates(df, FEATURES, group_col="mchID",
                         signal_col="IsSignal",
                         matchlabel_col="MatchLabel",
                         dummy_flag_col="is_dummy"
                         ):
    """
    Add one dummy candidate per group for ranking with 'no match' handling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with candidates.
    FEATURES : list
        List of feature column names used for training.
    group_col : str
        Column defining groups (e.g. mchID).
    signal_col : str
        Column indicating true match (1) vs background (0).
    matchlabel_col : str
        Column used for evaluation labeling.
    dummy_flag_col : str
        Name of dummy indicator column.
    Returns
    -------
    pd.DataFrame
        DataFrame with dummy candidates added.
    """

    # --- 2. Prepare dummy rows in a vectorized way
    excluded_cols = {dummy_flag_col, signal_col, matchlabel_col, group_col} # columns to not be set to 0
    feature_cols = [feat for feat in FEATURES if feat not in excluded_cols] # columns to be set to 0

    grouped = df.groupby(group_col, sort=False)
    group_ids = grouped.size().index
    has_signal = grouped[signal_col].any().to_numpy()

    df_dummy = pd.DataFrame(0.0, index=np.arange(len(group_ids)), columns=feature_cols)
    df_dummy.insert(0, group_col, group_ids)
    df_dummy[dummy_flag_col] = 1
    df_dummy[signal_col] = np.where(has_signal, 0, 1).astype(np.int8)
    df_dummy[matchlabel_col] = 8

    # --- 3. Concatenate
    df_out = pd.concat([df, df_dummy], ignore_index=True)

    return df_out

def perform_cuts(df: pd.DataFrame) -> pd.DataFrame:

    # Eta cuts TODO: confirm we do not pick up any illegal eta entries
    # eta_mch = np.arcsinh(pd.to_numeric(df["TanlMCH"], errors="raise"))
    # eta_mft = np.arcsinh(pd.to_numeric(df["TanlMFT"], errors = "raise"))
    # # Previously -2.45., adapted to reflect datamaker's limits for both MCH and MFT tracks
    # eta_mask = (eta_mch > -3.6) & (eta_mch < -2.5) & (eta_mft > -3.6) & (eta_mft < -2.5)
    # removed = df[~eta_mask].copy()
    # r_rows = int(removed.shape[0])
    # r_sig  = int(pd.to_numeric(removed.get("IsSignal", 0), errors="raise").sum())
    # r_bkg  = r_rows - r_sig
    # print("[Eta window] -4.0 < eta_MCH < -2.5 AND -3.6 < eta_MFT < -2.5")
    # print(f"Removed rows: {r_rows}  signal={r_sig}  background={r_bkg}")
    # df = df[eta_mask].reset_index(drop=True)
    
    # TODO: Repeat for other garbage values spotted

    # Drop garbage MFT entries TODO
    # mft_mask = (df['Chi2MFT'] < 1000)
    # removed = df[~mft_mask].copy()
    # r_rows = int(removed.shape[0])
    # r_sig  = int(pd.to_numeric(removed.get("IsSignal", 0), errors="coerce").sum())
    # r_bkg  = r_rows - r_sig
    # print(f"Removed rows with above 1000 MFT chi2: {r_rows}  signal={r_sig}  background={r_bkg}")
    # df = df[mft_mask].reset_index(drop=True)

    # drop rows with negative MFT variances (GARBAGE) TODO: Confirm with Andrea how we should approach this?
    var_mask = (df['CXXMFT'] > 0) & (df['CYYMFT'] > 0) & (df['CPhiPhiMFT'] > 0) & (df['CTglTglMFT'] > 0) & (df['C1Pt1PtMFT'] > 0)
    removed_rows = int((~var_mask).sum())
    r_sig = int(pd.to_numeric(df.loc[~var_mask, "IsSignal"], errors="coerce").sum())
    r_bkg = removed_rows - r_sig
    print(f"Removed rows with non-positive MFT variances: {removed_rows}  signal={r_sig}  background={r_bkg}")
    df = df.loc[var_mask].reset_index(drop=True)

    # wrap mft phi to [-pi, pi] --- right way to go about it, the outside of -pi->pi values are for the MFT tracks that take a helical path
    df['PhiMFT'] = np.arctan2(np.sin(df['PhiMFT']), np.cos(df['PhiMFT']))

    return df

def add_null_rows_for_non_pairable(df: pd.DataFrame) -> pd.DataFrame:
    # Identify pairable mchIDs
    pairable_mchIDs = df.groupby("mchID")["IsSignal"].any()

    # Create a DataFrame of non-pairable mchIDs with NaN values
    non_pairable_df = pd.DataFrame({
        "mchID": pairable_mchIDs[~pairable_mchIDs].index,
        "IsSignal": 0,  # or np.nan if you prefer
        "score": np.nan,  # or some default value
        # Add other columns as needed, filled with NaN or defaults
    })

    # Concatenate the original DataFrame with the non-pairable DataFrame
    df_full = pd.concat([df, non_pairable_df], ignore_index=True)

    return df_full



def inhousemetrics(
    df: pd.DataFrame,
    threshold: float = 0.5,
    metric: str = "score",
    Nsigma: float = 3.0,
) -> pd.DataFrame:
    idx = df.groupby("mchID")[metric].idxmax()
    best = df.loc[idx].set_index("mchID")

    #TODO: Revise the dummy candidates configuration
    # TODO: revise pairable definition - this still includes wrongs in pairable
    pairable = (
        df["MatchLabel"].isin(MATCH_LABEL_GROUPS["True"] + MATCH_LABEL_GROUPS["Wrong"])
        & (df["is_dummy"] == 0)
    ).groupby(df["mchID"]).any() 

    FakeNMissing = ~(
        ((df["IsSignal"] == 1) & (df["is_dummy"] == 0))
        .groupby(df["mchID"])
        .any()
    )

    is_reconstructed = (best[metric] > threshold) & (best["is_dummy"] == 0)
    # --- true match correctly reconstructed ---
    is_true = best["IsSignal"] == 1 # a bit debatable since this includes the dummy candidates
    is_true_reconstructed = is_reconstructed & is_true
    is_rejected = (best[metric] <= threshold) | (best["is_dummy"] == 1)

    N_total = len(best)
    N_pairable = pairable.sum()
    # N_non_pairable = N_total - N_pairable
    N_FakeNMissing = FakeNMissing.sum()
    N_gm_rec = is_reconstructed.sum()
    N_gm_true = is_true_reconstructed.sum()


    N_gm_rec_pairable = (is_reconstructed & pairable).sum()
    N_rejected_FakeNMissing = (is_rejected & FakeNMissing).sum()


    # --- Define metrics as (num, den) ---
    metrics = {
        "Purity": (N_gm_true, N_gm_rec),
        "Rec pairing efficiency": (N_gm_rec_pairable, N_pairable),
        "True pairing efficiency": (N_gm_true, N_pairable),
        "Wrong pairing fraction": (N_gm_rec_pairable - N_gm_true, N_pairable),
        "Rejection efficiency": (N_rejected_FakeNMissing, N_FakeNMissing),
    }

    rows = []

    for name, (num, den) in metrics.items():

        if den > 0:
            val = num / den
            err = Nsigma * np.sqrt(val * (1 - val) / den)
        else:
            val, err = np.nan, np.nan

        rows.append({
            "metric": name,
            "value": val,
            "uncertainty": err,
            "num": num,
            "den": den,
        })

    return pd.DataFrame(rows)



def plot_metrics_vs_feature(
    df: pd.DataFrame,
    feature: str,
    threshold: float,
    metrics_fn,
    metric_col_prefix: str = "score",
    bins: Optional[Union[int, np.ndarray]] = 10,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    trim_low: float = 0.0,
    trim_high: float = 0.0,
    Nsigma: float = 3.0,
):

    df = df.copy()

    # --- Optional trimming ---
    if trim_low > 0 or trim_high > 0:
        low_q = df[feature].quantile(trim_low)
        high_q = df[feature].quantile(1 - trim_high)
        df = df[(df[feature] >= low_q) & (df[feature] <= high_q)]

    # --- Range ---
    if fmin is None:
        fmin = df[feature].min()
    if fmax is None:
        fmax = df[feature].max()

    # --- Bin definition ---
    if isinstance(bins, int):
        edges = np.linspace(fmin, fmax, bins + 1)
    else:
        edges = np.asarray(bins)

    all_results = []

    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]

        if i == len(edges) - 2:
            df_bin = df[(df[feature] >= low) & (df[feature] <= high)]
        else:
            df_bin = df[(df[feature] >= low) & (df[feature] < high)]

        if len(df_bin) == 0:
            continue

        df_metrics = metrics_fn(
            df_bin,
            threshold=threshold,
            metric=metric_col_prefix,
            Nsigma=Nsigma,
        )

        # --- attach bin info ---
        df_metrics["bin_low"] = low
        df_metrics["bin_high"] = high
        df_metrics["bin_center"] = 0.5 * (low + high)
        df_metrics["bin_width"] = 0.5 * (high - low)
        df_metrics["entries"] = len(df_bin)

        all_results.append(df_metrics)

    result_df = pd.concat(all_results, ignore_index=True)

    # --- Plot ---
    plt.figure(figsize=(9, 6))

    for metric, subdf in result_df.groupby("metric"):

        # Skip broken metrics
        if subdf["value"].isna().all():
            print(f"[WARN] {metric} is all NaN → skipped")
            continue

        plt.errorbar(
            subdf["bin_center"],
            subdf["value"],
            yerr=subdf["uncertainty"],
            xerr=subdf["bin_width"],
            fmt='o',
            capsize=3,
            label=metric,
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
    **kwargs,
) -> None:
    """
    Plot a histogram, normalised or not, (continuous) or grouped bar chart (categorical)
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
    non_empty_groups = [g for g in match_groups.values() if len(g) > 0]
    
    if not non_empty_groups:
        # All groups are empty; skip this feature
        return
    
    sample_col = non_empty_groups[0][feature]
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
                .value_counts(normalize=density) # Added density instead of defaulting to True for some reason - now have count based barplots
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
        # for label, g in match_groups.items():
        #     if len(g) > 0:
        #         qmin = g[feature].quantile(per)
        #         qmax = g[feature].quantile(1 - per)
        #         print(f"{label:15s}: {qmin:.3g} → {qmax:.3g}")
        minn = min(g[feature].quantile(per) for g in match_groups.values() if len(g) > 0)
        maxx = max(g[feature].quantile(1 - per) for g in match_groups.values() if len(g) > 0)
        
        # Avoid degenerate histogram range (minn == maxx) — expand slightly
        if minn == maxx:
            epsilon = 1e-6 if abs(minn) < 1 else abs(minn) * 1e-6
            minn -= epsilon
            maxx += epsilon
        
        for label, group in match_groups.items():
            if len(group) == 0:
                continue  # Skip empty groups to avoid histogram warnings
            # if len(group) > 0:
            #     print(label, group[feature].quantile(per), group[feature].quantile(1 - per))
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
                **kwargs
            )
    ax.set_ylabel("Density" if density else "Counts", fontsize=20, labelpad=15)
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


def sweep_threshold_plot(
    df_eval: pd.DataFrame,
    metrics_fn,
    score_col: str = "score",
    n_steps: int = 100,
    title: str = "Metrics vs Threshold",
    Nsigma: float = 1.0,
):
    """
    Sweep threshold and plot metrics using structured DataFrame output.
    """

    score_min = df_eval[score_col].min()
    score_max = df_eval[score_col].max()
    thresholds = np.linspace(score_min, score_max, n_steps)

    all_results = []

    for t in thresholds:
        df_metrics = metrics_fn(
            df_eval,
            threshold=t,
            metric=score_col,
            Nsigma=Nsigma,
        )

        df_metrics["threshold"] = t
        all_results.append(df_metrics)
    print(f"Computed metrics are {all_results} thresholds.")

    result_df = pd.concat(all_results, ignore_index=True)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))

    for metric, subdf in result_df.groupby("metric"):

        if subdf["value"].isna().all():
            print(f"[WARN] {metric} is all NaN → skipped")
            continue

        # Sort for clean lines
        subdf = subdf.sort_values("threshold")

        ax.plot(
            subdf["threshold"],
            subdf["value"],
            lw=2,
            label=metric,
        )

        # --- Optional uncertainty band ---
        if "uncertainty" in subdf.columns:
            ax.fill_between(
                subdf["threshold"],
                subdf["value"] - subdf["uncertainty"],
                subdf["value"] + subdf["uncertainty"],
                alpha=0.2,
            )

    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return result_df