# Common utilities for the matching - used in both EDA & matching code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hipe4ml.tree_handler import TreeHandler
from typing import Optional, Union


#TODO: Add some metric of total uncertainty... like something derived of the whole covariance matrix...that's dangerously close to CHI2 again.
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

NON_TRAINING_FEATURES = [ # features taht are unsuitable for training, but are still read in for other uses like labelling and analysis
    'mchID',
    'MatchLabel', 'IsSignal',
    # features we exclude based on bias & selection cuts, i.e. we do not want our model to discriminate based on these, since it learns what we tell it to, not what we intend for it to
    # Otherwise we risk it learning for example that non-prompts are bad
    # Interpretation still pending
    'DCAX', 'DCAY', 'PDCA', 'DCAXY', 'Rabs', 'IsAmbig', 'Chi2MCH', 'Chi2MFT', 'MFTMult', 'MatchAttempts',
    ]


ALL_FEATURES = ['fXMCH', 'fYMCH', 'fPhiMCH', 'fTanlMCH', 'fInvQPtMCH', 'fTimeMCH',
       'fTimeResMCH', 'fChi2MCH', 'fPDCA', 'fRabs', 'fCXXMCH', 'fCYYMCH',
       'fCPhiPhiMCH', 'fCTglTglMCH', 'fC1Pt1PtMCH', 'fCXYMCH', 'fCPhiYMCH',
       'fCPhiXMCH', 'fCTglXMCH', 'fCTglYMCH', 'fCTglPhiMCH', 'fC1PtXMCH',
       'fC1PtYMCH', 'fC1PtPhiMCH', 'fC1PtTglMCH', 'fXMFT', 'fYMFT', 'fPhiMFT',
       'fTanlMFT', 'fInvQPtMFT', 'fTimeMFT', 'fTimeResMFT', 'fChi2MFT',
       'fMftClusterSizesAndTrackFlags', 'fTrackTypeMFT', 'fCXXMFT', 'fCYYMFT',
       'fCPhiPhiMFT', 'fCTglTglMFT', 'fC1Pt1PtMFT', 'fCXYMFT', 'fCPhiYMFT',
       'fCPhiXMFT', 'fCTglXMFT', 'fCTglYMFT', 'fCTglPhiMFT', 'fC1PtXMFT',
       'fC1PtYMFT', 'fC1PtPhiMFT', 'fC1PtTglMFT', 'fChi2Glob', 'fChi2Match',
       'fDCAX', 'fDCAY', 'fIsAmbig', 'fMFTMult', 'fMatchAttempts',
       'fMcMaskMCH', 'fMcMaskMFT', 'fMcMaskGlob', 'fMatchLabel', 'fIsSignal']

SKIPPED_FEATURES  = [
    'fTimeMCH', 'fTimeResMCH', 'fTimeMFT', 'fTimeResMFT', 
    'fMftClusterSizesAndTrackFlags', 
    'fChi2Glob', 'fChi2Match',
    'fMcMaskMCH', 'fMcMaskMFT', 'fMcMaskGlob',
    ]

READ_FEATURES = [f for f in ALL_FEATURES if f not in SKIPPED_FEATURES]

#NOTE: Once we run into issues with overlaps in MCHID we can consider using more features to define the group, for now this is sufficient
GROUP_PRESERVING_FEATURES = [
    'XMCH', 'YMCH', 'PhiMCH', 'TanlMCH', 'InvQPtMCH',# "chi2MCH",
    'Chi2MCH', 'PDCA', 'Rabs', 'CXXMCH', 'CYYMCH', 'CPhiPhiMCH', 'CTglTglMCH', 'C1Pt1PtMCH',
    'MatchAttempts', 'MFTMult', 'PtMCH',
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
    df = TreeHandler(file_path, "O2fwdmlcand", folder_name=folder_name, column_names=READ_FEATURES).get_data_frame()
    df.columns = df.columns.str.replace(r'^f', '', regex=True) # Drop leading 'f'
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        print("converting bools to ints for columns:", bool_cols.tolist())
        df[bool_cols] = df[bool_cols].astype(int)
    return df

def process_dataframe(df: pd.DataFrame, makedummies: bool = False) -> pd.DataFrame:
    df = design_features(df)
    print(f"After feature design, shape: {df.shape}")

    df = perform_cuts(df) 
    print(f"After cuts, shape: {df.shape}")

    # --- 3. Add dummy candidates for non-pairable groups ---
    if makedummies:
        df = add_dummy_candidates(df, FEATURES=[f for f in df.columns.tolist() if f not in NON_TRAINING_FEATURES], group_col="mchID", signal_col="IsSignal", matchlabel_col="MatchLabel", dummy_flag_col="is_dummy")
        print(f"Added dummy candidates. New shape: {df.shape}")
    
    return df


def design_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df['PhiMFT'] = np.arctan2(np.sin(df['PhiMFT']), np.cos(df['PhiMFT']))

    df['etaMCH'] = np.arcsinh(df['TanlMCH']).astype(np.float32)
    df['etaMFT'] = np.arcsinh(df['TanlMFT']).astype(np.float32)
    df['DeltaEta'] = (df['etaMCH'] - df['etaMFT']).astype(np.float32)

    df['DCAXY'] = np.sqrt(df['DCAX']**2 + df['DCAY']**2).astype(np.float32)

    # df["is_dummy"] = 0 # terminated for now

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
    df['PtMCH'] = (1 / np.abs(df['InvQPtMCH'])).astype(np.float32)
    df['PtMFT'] = (1 / np.abs(df['InvQPtMFT'])).astype(np.float32)
    df['DeltaPt'] = (df['PtMCH'] - df['PtMFT']).astype(np.float32)
    df['RelPtDiff'] = (df['DeltaPt'] / (df['PtMFT'] + df['PtMCH'])).astype(np.float32) # relative pt difference
    df['PullPt'] = (df['DeltaPt'] / np.sqrt(df['C1Pt1PtMCH']/df['InvQPtMCH']**4 + df['C1Pt1PtMFT']/df['InvQPtMFT']**4)).astype(np.float32)
    # TODO: formula seems appropriate, but the plot for the pull is non-gaussian

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
    """
    Apply the MFT preselection cuts with minimal temporary allocation.

    The implementation works directly on NumPy views of the relevant columns,
    avoids materialising intermediate DataFrames for the removed rows,
    and performs the final row filtering only once.
    """

    # Use NumPy views for the hot columns; this avoids repeated pandas object
    # dispatch overhead on a very large dataframe.
    signal_values = df["IsSignal"].to_numpy(dtype=np.int8, copy=False)
    eta_vals = df["etaMFT"].to_numpy(dtype=np.float32, copy=False)
    chi2_vals = df["Chi2MFT"].to_numpy(dtype=np.float32, copy=False)
    cxx_vals = df["CXXMFT"].to_numpy(dtype=np.float32, copy=False)
    cyy_vals = df["CYYMFT"].to_numpy(dtype=np.float32, copy=False)
    cphi_vals = df["CPhiPhiMFT"].to_numpy(dtype=np.float32, copy=False)
    ctgl_vals = df["CTglTglMFT"].to_numpy(dtype=np.float32, copy=False)
    c1pt_vals = df["C1Pt1PtMFT"].to_numpy(dtype=np.float32, copy=False)

    # --- 1) Loose eta window ---
    eta_mask = (eta_vals >= -3.7) & (eta_vals <= -2.4)
    removed_eta_rows = int((~eta_mask).sum())
    removed_eta_sig = int(signal_values[~eta_mask].sum())
    removed_eta_bkg = removed_eta_rows - removed_eta_sig
    print("[Loose Eta window] -3.7 < eta_MFT < -2.4")
    print(
        f"Removed rows: {removed_eta_rows}  signal={removed_eta_sig}  background={removed_eta_bkg}"
    )

    # --- 2) Garbage MFT chi2 entries ---
    mft_mask = chi2_vals < 1000.0
    eta_mft_mask = eta_mask & mft_mask
    removed_mft_rows = int((eta_mask & ~mft_mask).sum())
    removed_mft_sig = int(signal_values[eta_mask & ~mft_mask].sum())
    removed_mft_bkg = removed_mft_rows - removed_mft_sig
    print(f"Removed rows with above 1000 MFT chi2: {removed_mft_rows}  signal={removed_mft_sig}  background={removed_mft_bkg}")

    # --- 3) Non-positive MFT variances ---
    var_mask = (
        (cxx_vals > 0)
        & (cyy_vals > 0)
        & (cphi_vals > 0)
        & (ctgl_vals > 0)
        & (c1pt_vals > 0)
    )
    final_mask = eta_mft_mask & var_mask
    removed_var_rows = int((eta_mft_mask & ~var_mask).sum())
    removed_var_sig = int(signal_values[eta_mft_mask & ~var_mask].sum())
    removed_var_bkg = removed_var_rows - removed_var_sig
    print(
        f"Removed rows with non-positive MFT variances: {removed_var_rows}  signal={removed_var_sig}  background={removed_var_bkg}"
    )

    return df.loc[final_mask].reset_index(drop=True)

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

    # Terminated Dummies for now
    # TODO: revise pairable definition - this still includes wrongs in pairable
    # Optionally we can add a configurable on if we should allows us to tweak if we include missing matches in pairable or not.
    # This is considered an irreducible errorr
    # In the realm of ~1-5% for OO vs PbPb
    # Need to decide on this at some point, depends on if we want to asess the model's maximal achievable performance or the performance on real data....
    pairable = (
        df["MatchLabel"].isin(MATCH_LABEL_GROUPS["True"] + MATCH_LABEL_GROUPS["Wrong"])
        # & (df["is_dummy"] == 0) # Ensures that dummies are not included in our definition of pairable... but we do include missing?... metrics are due for a revision
    ).groupby(df["mchID"]).any() 

    FakeNMissing = ~(
        ((df["IsSignal"] == 1)
        #  & (df["is_dummy"] == 0)
         )
        .groupby(df["mchID"])
        .any()
    )

    is_reconstructed = (best[metric] > threshold) #& (best["is_dummy"] == 0)
    # --- true match correctly reconstructed ---
    is_true = best["IsSignal"] == 1 # a bit debatable since this includes the dummy candidates
    is_true_reconstructed = is_reconstructed & is_true
    is_rejected = (best[metric] <= threshold) #| (best["is_dummy"] == 1)

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
    fig = go.Figure()

    for metric, subdf in result_df.groupby("metric"):
        if subdf["value"].isna().all():
            print(f"[WARN] {metric} is all NaN → skipped")
            continue

        fig.add_trace(
            go.Scatter(
                x=subdf["bin_center"],
                y=subdf["value"],
                error_y=dict(
                    type="data",
                    array=subdf["uncertainty"],
                    visible=True,
                ),
                mode="markers+lines",
                name=metric,
            )
        )

    fig.update_layout(
        title=f"Metrics vs {feature} (threshold={threshold})",
        xaxis_title=feature,
        yaxis_title="Metric",
        template="plotly_white",
    )

    fig.show()

    return result_df


def plot_metrics_vs_xy(
    df: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    threshold: float,
    metrics_fn,
    metric_col_prefix: str = "score",
    x_bins: Optional[Union[int, np.ndarray]] = 10,
    y_bins: Optional[Union[int, np.ndarray]] = 10,
    fmin_x: Optional[float] = None,
    fmax_x: Optional[float] = None,
    fmin_y: Optional[float] = None,
    fmax_y: Optional[float] = None,
    trim_low: float = 0.0,
    trim_high: float = 0.0,
    Nsigma: float = 3.0,
    cmap: str = "viridis",
    surface_alpha: float = 0.9,
):
    """Plot inhouse metrics as a function of two features.

    The function computes metrics in 2D bins over (feature_x, feature_y) and
    creates a separate surface plot for each metric returned by `metrics_fn`.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and score.
    feature_x : str
        Column name for the x axis.
    feature_y : str
        Column name for the y axis.
    threshold : float
        Score threshold passed to `metrics_fn`.
    metrics_fn : callable
        A function like `inhousemetrics` that returns a dataframe of metric
        rows with columns ['metric', 'value', 'uncertainty', 'num', 'den'].
    metric_col_prefix : str
        Score column name used by `metrics_fn`.
    x_bins, y_bins : int or array-like
        Number of bins or explicit bin edges for the x and y axes.
    fmin_x, fmax_x, fmin_y, fmax_y : float
        Optional axis bounds. Defaults to the data range.
    trim_low, trim_high : float
        Optional quantile trimming for both axes.
    Nsigma : float
        Uncertainty band scaling passed to `metrics_fn`.
    cmap : str
        Matplotlib colormap for the surface.
    surface_alpha : float
        Opacity of the surface plot.
    """

    df = df.copy()

    if trim_low > 0 or trim_high > 0:
        x_low_q = df[feature_x].quantile(trim_low)
        x_high_q = df[feature_x].quantile(1 - trim_high)
        y_low_q = df[feature_y].quantile(trim_low)
        y_high_q = df[feature_y].quantile(1 - trim_high)
        df = df[
            (df[feature_x] >= x_low_q)
            & (df[feature_x] <= x_high_q)
            & (df[feature_y] >= y_low_q)
            & (df[feature_y] <= y_high_q)
        ]

    if fmin_x is None:
        fmin_x = df[feature_x].min()
    if fmax_x is None:
        fmax_x = df[feature_x].max()
    if fmin_y is None:
        fmin_y = df[feature_y].min()
    if fmax_y is None:
        fmax_y = df[feature_y].max()

    def _bin_edges(name, bins, vmin, vmax):
        if isinstance(bins, (np.ndarray, list)):
            return np.asarray(bins, dtype=float)
        if isinstance(bins, int):
            return np.linspace(vmin, vmax, bins + 1)
        raise ValueError(f"{name} must be int or array-like")

    x_edges = _bin_edges("x_bins", x_bins, fmin_x, fmax_x)
    y_edges = _bin_edges("y_bins", y_bins, fmin_y, fmax_y)

    all_results = []

    for ix in range(len(x_edges) - 1):
        x_low, x_high = x_edges[ix], x_edges[ix + 1]
        x_center = 0.5 * (x_low + x_high)

        for iy in range(len(y_edges) - 1):
            y_low, y_high = y_edges[iy], y_edges[iy + 1]
            y_center = 0.5 * (y_low + y_high)

            if ix == len(x_edges) - 2:
                mask_x = (df[feature_x] >= x_low) & (df[feature_x] <= x_high)
            else:
                mask_x = (df[feature_x] >= x_low) & (df[feature_x] < x_high)

            if iy == len(y_edges) - 2:
                mask_y = (df[feature_y] >= y_low) & (df[feature_y] <= y_high)
            else:
                mask_y = (df[feature_y] >= y_low) & (df[feature_y] < y_high)

            df_bin = df[mask_x & mask_y]
            if len(df_bin) == 0:
                continue

            df_metrics = metrics_fn(
                df_bin,
                threshold=threshold,
                metric=metric_col_prefix,
                Nsigma=Nsigma,
            )

            df_metrics = df_metrics.copy()
            df_metrics["bin_low_x"] = x_low
            df_metrics["bin_high_x"] = x_high
            df_metrics["bin_center_x"] = x_center
            df_metrics["bin_low_y"] = y_low
            df_metrics["bin_high_y"] = y_high
            df_metrics["bin_center_y"] = y_center
            df_metrics["entries"] = len(df_bin)
            all_results.append(df_metrics)

    if len(all_results) == 0:
        raise ValueError("No data available in the requested x/y binning range.")

    result_df = pd.concat(all_results, ignore_index=True)

    metrics = result_df["metric"].unique()
    n_metrics = len(metrics)
    ncols = min(2, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))

    zmin = result_df["value"].min()
    zmax = result_df["value"].max()

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=list(metrics),
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
        specs=[[{"type": "xy"} for _ in range(ncols)] for _ in range(nrows)],
    )

    first_colorbar = True

    for metric_idx, metric in enumerate(metrics):
        row = metric_idx // ncols + 1
        col = metric_idx % ncols + 1
        subdf = result_df[result_df["metric"] == metric]

        x_centers = np.sort(subdf["bin_center_x"].unique())
        y_centers = np.sort(subdf["bin_center_y"].unique())

        Z = subdf.pivot(
            index="bin_center_y",
            columns="bin_center_x",
            values="value",
        ).reindex(index=y_centers, columns=x_centers).to_numpy()

        den = subdf.pivot(
            index="bin_center_y",
            columns="bin_center_x",
            values="den",
        ).reindex(index=y_centers, columns=x_centers).to_numpy()

        num = subdf.pivot(
            index="bin_center_y",
            columns="bin_center_x",
            values="num",
        ).reindex(index=y_centers, columns=x_centers).to_numpy()

        entries = subdf.pivot(
            index="bin_center_y",
            columns="bin_center_x",
            values="entries",
        ).reindex(index=y_centers, columns=x_centers).to_numpy()

        customdata = np.stack([den, num, entries], axis=-1)

        colorbar = dict(title="Value", len=0.8, yanchor="middle", y=0.5, x=1.02)
        if not first_colorbar:
            colorbar = None

        heatmap_kwargs = {
            "x": x_centers,
            "y": y_centers,
            "z": Z,
            "colorscale": cmap,
            "zmin": zmin,
            "zmax": zmax,
            "zsmooth": "best",
            "customdata": customdata,
            "hovertemplate": (
                f"{feature_x}: %{{x:.3f}}<br>"
                f"{feature_y}: %{{y:.3f}}<br>"
                "Value: %{z:.3f}<br>"
                "Num: %{customdata[1]}<br>"
                "Den: %{customdata[0]}<br>"
                "Entries: %{customdata[2]}<extra>" + metric + "</extra>"
            ),
            "showscale": first_colorbar,
            "showlegend": False,
        }
        if colorbar is not None:
            heatmap_kwargs["colorbar"] = colorbar

        fig.add_trace(
            go.Heatmap(**heatmap_kwargs),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text=feature_x, row=row, col=col)
        fig.update_yaxes(title_text=feature_y, row=row, col=col)

        first_colorbar = False

    fig.update_layout(
        title_text=f"Metrics vs {feature_x} and {feature_y} (threshold={threshold})",
        template="plotly_white",
        height=450 * nrows,
        width=700 * ncols,
        showlegend=False,
    )

    fig.show()

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