# Common utilities for the matching - used in both EDA & matching code
import pandas as pd
import numpy as np
from hipe4ml.tree_handler import TreeHandler


DESIGNED_FEATURES = [
    "mchID",
    # Deltas
    'DeltaX', 'DeltaY', 'DeltaPhi', 'DeltaTanl', 'SameSign', 'PT',
    # Pulls
    'PullX', 'PullY', 'PullPhi', 'PullTanl',
    'DeltaDirection' # Angle mismatch between MCH and MFT tracks, calculated as the angle between their momentum vectors
    ]


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
    df['RelPTDiff'] = (1/np.abs(invqptmch) - 1/np.abs(invqptmft))/(1/np.abs(invqptmch)+1/np.abs(invqptmft)) # relative curvature differene

    df['SameSign'] = (np.signbit(invqptmch) == np.signbit(invqptmft)).astype(np.int8)
    df['PT'] = 1 / np.abs(invqptmch) # Rocking only with the MCH Pt for now - gives a consistent value for eventual binning procedure

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

#.any.sum practically counts the number of unique mchIDs that have at least one true
def inhousemetrics(df: pd.DataFrame, threshold: float = 0.5) -> tuple:
    idx = df.groupby("mchID")["score"].idxmax() # max score index in base df for each mchID group
    best = df.loc[idx].set_index("mchID") # best candidate for each mchID, indexed by mchID
    pairable = df.groupby("mchID")["IsSignal"].any() # Boolean series indicating if each mchID group has at least one true match - indexed by mchID

    

    N_pairable = pairable.sum()

    # N_non_pairable = 

    N_gm_rec = (df.loc[idx, "score"] > threshold).sum()

    N_gm_true = (
        (best["score"] > threshold) &
        (best["IsSignal"] == 1)
    ).sum()

    N_gm_rec_pairable = (
        (best["score"] > threshold) &
        pairable
    ).sum()

    N_gm_true_pairable = N_gm_true  # already pairable by construction ---kind of trivializes a bit

    pairing_purity = N_gm_true/N_gm_rec if N_gm_rec > 0 else 0
    pairing_efficiency = N_gm_rec_pairable/N_pairable if N_pairable > 0 else 0
    true_efficiency = N_gm_true_pairable/N_pairable if N_pairable > 0 else 0
    fake_efficiency = (N_gm_rec_pairable - N_gm_true_pairable)/N_pairable if N_pairable > 0 else 0
    # fake_efficiency_new_nomenclature = (N_gm_rec_pairable - N_gm_true_pairable)/(len(df.groupby("mchID").len())-N_pairable) if N_pairable > 0 else 0 # N_gm_rec - N_gm_rec_pairable = N_gm_nonpairable /
    # Pivot to standard new nomenclature


    return pairing_purity, pairing_efficiency, true_efficiency, fake_efficiency