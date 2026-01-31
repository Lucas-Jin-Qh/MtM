# pyre-strict

import os 
from typing import Any, Optional, Tuple, Dict, List, Union

from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import logging

from iblutil.numerical import ismember
from iblatlas.regions import BrainRegions
from brainbox.io.one import (
    SpikeSortingLoader, 
    SessionLoader
)

from src.constants import DISCRETE_BEHAVIOR, CONTINUOUS_BEHAVIOR

logger = logging.getLogger(__name__)

import uuid
import sys
import multiprocessing
from tqdm import tqdm
from functools import partial
from scipy.interpolate import interp1d
from iblutil.numerical import ismember, bincount2D
from brainbox.population.decode import get_spike_counts_in_bins

def globalize(func):
  def result(*args, **kwargs):
    return func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result

def load_spiking_data(
    one: Any,
    pid: str,
    compute_metrics: bool = False,
    qc: Optional[int] = None,
    *,
    eid: str = "",
    pname: str = "",
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[pd.DataFrame], float]:
        
    spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    # Try to get sampling frequency from raw electrophysiology (AP band)
    try:
        sampling_freq = spike_loader.raw_electrophysiology(band="ap", stream=True).fs
        sampling_freq = float(sampling_freq) if sampling_freq is not None else 30000.0
    except Exception:
        sampling_freq = 30000.0

    spikes, clusters, channels = spike_loader.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=compute_metrics
    )

    if clusters_labeled is None:
        return None, None, sampling_freq
    clusters_labeled_df = clusters_labeled.to_df()

    if qc is None:
        return spikes, clusters_labeled_df, sampling_freq

    iok = clusters_labeled_df["label"] >= qc
    selected_clusters = clusters_labeled_df[iok]
    spike_idx, ib = ismember(spikes["clusters"], selected_clusters.index.to_numpy())
    selected_clusters.reset_index(drop=True, inplace=True)
    selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
    selected_spikes["clusters"] = selected_clusters.index.to_numpy()[ib].astype(np.int32)
    return selected_spikes, selected_clusters, sampling_freq


def merge_probes(
    spikes_list: List[Dict[str, np.ndarray]],
    clusters_list: List[pd.DataFrame],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    assert len(clusters_list) == len(spikes_list), "Mismatched list lengths"
    assert all(isinstance(s, dict) for s in spikes_list)
    assert all(isinstance(c, pd.DataFrame) for c in clusters_list)

    merged_spikes: List[Dict[str, np.ndarray]] = []
    merged_clusters: List[pd.DataFrame] = []
    cluster_max: int = 0

    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes["clusters"] += cluster_max
        cluster_max = clusters.index.max() + 1
        merged_spikes.append(spikes)
        merged_clusters.append(clusters)

    merged_clusters_df = pd.concat(merged_clusters, ignore_index=True)
    merged_spikes_dict = {
        k: np.concatenate([s[k] for s in merged_spikes]) for k in merged_spikes[0].keys()
    }
    sort_idx = np.argsort(merged_spikes_dict["times"], kind="stable")
    merged_spikes_sorted = {k: v[sort_idx] for k, v in merged_spikes_dict.items()}
    return merged_spikes_sorted, merged_clusters_df


def load_trials_and_mask(
    one: Any,
    eid: str,
    min_rt: float = 0.0,
    max_rt: float = 10.0,
    nan_exclude: Union[str, List[str]] = "default",
    min_trial_len: Optional[float] = None,
    max_trial_len: Optional[float] = 10.0,
    exclude_unbiased: bool = False,
    exclude_nochoice: bool = True,
    sess_loader: Optional[SessionLoader] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    if nan_exclude == "default":
        nan_exclude = [
            "stimOn_times",
            "choice",
            "feedback_times",
            "probabilityLeft",
            "firstMovement_times",
            "feedbackType",
        ]

    if sess_loader is None:
        sess_loader = SessionLoader(one=one, eid=eid)  # Create session loader for IBL data access

    if sess_loader.trials.empty:
        sess_loader.load_trials()

    query = ""
    if min_rt is not None:
        query += f"(firstMovement_times - stimOn_times < {min_rt})"
    if max_rt is not None:
        query += f" | (firstMovement_times - stimOn_times > {max_rt})"
    if min_trial_len is not None:
        query += f" | (feedback_times - goCue_times < {min_trial_len})"
    if max_trial_len is not None:
        query += f" | (feedback_times - goCue_times > {max_trial_len})"
    for event in nan_exclude:
        query += f" | {event}.isnull()"
    if exclude_unbiased:
        query += " | (probabilityLeft == 0.5)"
    if exclude_nochoice:
        query += " | (choice == 0)"
    if min_rt is None and query.startswith(" | "):
        query = query[3:]

    mask = ~sess_loader.trials.eval(query)
    return sess_loader.trials, mask


def list_brain_regions(
    neural_dict: Dict[str, Any], *, single_region: bool = False
) -> Tuple[List[np.ndarray], np.ndarray]:
    brainreg = BrainRegions()
    beryl_reg = brainreg.acronym2acronym(neural_dict["cluster_regions"], mapping="Beryl")
    unique_regions = np.unique(beryl_reg)

    if single_region:
        regions = [np.array([region]) for region in unique_regions]
    else:
        regions = [unique_regions]

    return regions[0], beryl_reg


def select_brain_regions(
    beryl_reg: np.ndarray, region: np.ndarray
) -> np.ndarray:
    reg_mask = np.isin(beryl_reg, region)
    reg_clu_ids = np.argwhere(reg_mask).flatten()
    return reg_clu_ids


def create_intervals(
    start_time: float, end_time: float, interval_len: float
) -> np.ndarray:
    interval_begs = np.arange(start_time, end_time - interval_len, interval_len)
    interval_ends = np.arange(start_time + interval_len, end_time, interval_len)
    return interval_begs, interval_ends


def get_spike_data_per_interval(
    times, 
    clusters, 
    interval_begs, 
    interval_ends, 
    interval_len, 
    binsize, 
    n_workers=os.cpu_count()
):
    n_intervals = len(interval_begs)

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    cluster_ids = np.unique(clusters)
    n_clusters_in_region = len(cluster_ids)

    @globalize
    def compute_spike_count(interval):
        interval_idx, t_beg, t_end = interval
        idxs_t = (times >= t_beg) & (times < t_end)
        times_curr = times[idxs_t]
        clust_curr = clusters[idxs_t]
        if times_curr.shape[0] == 0:
            # no spikes in this trial
            binned_spikes_tmp = np.zeros((n_clusters_in_region, n_bins))
            if np.isnan(t_beg) or np.isnan(t_end):
                t_idxs = np.nan * np.ones(n_bins)
            else:
                t_idxs = np.arange(t_beg, t_end + binsize / 2, binsize)
            idxs_tmp = np.arange(n_clusters_in_region)
        else:
            # bin spikes
            binned_spikes_tmp, t_idxs, cluster_idxs = bincount2D(
                times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
            # find indices of clusters that returned spikes for this trial
            _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)
        return binned_spikes_tmp[:, :n_bins], idxs_tmp, interval_idx

    binned_spikes = np.zeros((n_intervals, n_clusters_in_region, n_bins))
    with multiprocessing.Pool(processes=n_workers) as p:
        intervals = list(zip(np.arange(n_intervals), interval_begs, interval_ends))
        with tqdm(total=len(intervals)) as pbar:
            for res in p.imap_unordered(compute_spike_count, intervals):
                pbar.update()
                binned_spikes[res[-1], res[1], :] += res[0]
        pbar.close()
        p.close()
    return binned_spikes


def bin_spiking_data(
    reg_clu_ids, 
    neural_df, 
    intervals=None, 
    trials_df=None, 
    n_workers=os.cpu_count(), 
    **kwargs
):
    if trials_df is not None:
        intervals = np.vstack([
            trials_df[kwargs["align_time"]] + kwargs["time_window"][0],
            trials_df[kwargs["align_time"]] + kwargs["time_window"][1]
        ]).T
        chunk_len = kwargs["time_window"][1] - kwargs["time_window"][0]
        interval_len = (
            kwargs["time_window"][1] - kwargs["time_window"][0]
        )
    else:
        assert intervals is not None, \
            "Require intervals to segment the recording into chunks including trials and non-trials."
        chunk_len = intervals[0,1] - intervals[0,0]
        interval_len = (
            intervals[0,1] - intervals[0,0]
        )
    # subselect spikes for this region
    spikemask = np.isin(neural_df["spike_clusters"], reg_clu_ids)
    regspikes = neural_df["spike_times"][spikemask]
    regclu = neural_df["spike_clusters"][spikemask]
    clusters_used_in_bins = np.unique(regclu)
    binsize = kwargs.get("binsize", chunk_len)
    
    if chunk_len / binsize == 1.0:
        # one vector of neural activity per interval
        binned, _ = get_spike_counts_in_bins(regspikes, regclu, intervals)
        binned = binned.T  # binned is a 2D array
        binned_list = [x[None, :] for x in binned]
    else:
        binned_array = get_spike_data_per_interval(
            regspikes, regclu,
            interval_begs=intervals[:, 0],
            interval_ends=intervals[:, 1],
            interval_len=interval_len,
            binsize=kwargs["binsize"],
            n_workers=n_workers
        )
        binned_list = [x.T for x in binned_array]   
    return np.array(binned_list), clusters_used_in_bins

def load_behavior_data(
    one: Any,
    eid: str,
    trials_df: pd.DataFrame,
    behavior_keys: Optional[List[str]] = None,
    use_raw_encoding: bool = True,  # If True, use original IBL encoding
) -> Dict[str, np.ndarray]:
    if behavior_keys is None:
        behavior_keys = DISCRETE_BEHAVIOR + CONTINUOUS_BEHAVIOR

    sess_loader = SessionLoader(one=one, eid=eid)  # Create session loader for accessing behavioral data

    behavior_data: Dict[str, np.ndarray] = {}
    for key in behavior_keys:
        if key in trials_df.columns:
            behavior_data[key] = trials_df[key].to_numpy()
        elif key == "block":
            # Use original probabilityLeft values for block (0.2, 0.5, 0.8)
            behavior_data[key] = trials_df["probabilityLeft"].to_numpy()
        elif key == "reward":
            behavior_data[key] = (trials_df["rewardVolume"] > 1).astype(int).to_numpy()
        elif key == "wheel":
            sess_loader.load_wheel()
            behavior_data[key] = {
                "times": sess_loader.wheel["times"].to_numpy(),
                "values": np.abs(sess_loader.wheel["velocity"].to_numpy()),
            }
        elif key == "left-whisker":
            sess_loader.load_motion_energy(views=["left"])
            behavior_data[key] = {
                "times": sess_loader.motion_energy["leftCamera"]["times"].to_numpy(),
                "values": sess_loader.motion_energy["leftCamera"]["whiskerMotionEnergy"].to_numpy(),
            }
        elif key == "right-whisker":
            sess_loader.load_motion_energy(views=["right"])
            behavior_data[key] = {
                "times": sess_loader.motion_energy["rightCamera"]["times"].to_numpy(),
                "values": sess_loader.motion_energy["rightCamera"]["whiskerMotionEnergy"].to_numpy(),
            }
        else:
            raise ValueError(f"Unknown behavior key: {key}")

    # Use original IBL encoding (choice: -1/1, block: 0.2/0.5/0.8)
    # Integer encoding conversion has been removed

    return behavior_data


def prepare_data(
    one: Any, eid: str, params: Dict[str, Any]
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    pids, probe_names = one.eid2pid(eid)
    details = one.get_details(eid)
    logger.info(f"Merge {len(probe_names)} probes for session EID: {eid}")

    clusters_list: List[pd.DataFrame] = []
    spikes_list: List[Dict[str, np.ndarray]] = []

    sampling_freq = None
    for pid, probe_name in zip(pids, probe_names):
        tmp_spikes, tmp_clusters, tmp_sampling_freq = load_spiking_data(
            one, pid, eid=eid, pname=probe_name
        )
        if tmp_spikes is None or tmp_clusters is None:
            return None, None, None, None
        if sampling_freq is None:
            sampling_freq = tmp_sampling_freq
        tmp_clusters["pid"] = pid
        spikes_list.append(tmp_spikes)
        clusters_list.append(tmp_clusters)

    spikes, clusters = merge_probes(spikes_list, clusters_list)

    trials_df, trials_mask = load_trials_and_mask(one=one, eid=eid)

    # Define brain parcellation
    brainreg = BrainRegions()
    cluster_regions = brainreg.acronym2acronym(
        clusters["acronym"].to_numpy(), mapping="Beryl"
    )

    neural_dict: Dict[str, Any] = {
        "spike_times": spikes["times"],
        "spike_clusters": spikes["clusters"],
    }

    trials_data: Dict[str, Any] = {
        "trials_df": trials_df,
        "trials_mask": trials_mask,
    }

    meta_data: Dict[str, Any] = {
        "eid": eid,
        "subject": details["subject"],
        "lab": details["lab"],
        "cluster_channels": list(clusters["channels"]),
        "cluster_regions": list(cluster_regions),
        "uuids": list(clusters["uuids"]),
        "cluster_depths": list(clusters["depths"]) if "depths" in clusters.columns else [],
        "good_clusters": list((clusters["label"] >= 1).astype(int)) if "label" in clusters.columns else [],
        "sampling_freq": float(sampling_freq) if sampling_freq is not None else 30000.0,
    }

    behavior_data = load_behavior_data(
        one, eid, trials_df,
        behavior_keys=params.get("behavior_keys", None),
        use_raw_encoding=params.get("use_raw_encoding", True),
    )

    return neural_dict, meta_data, trials_data, behavior_data

