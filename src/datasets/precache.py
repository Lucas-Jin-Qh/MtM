# pyre-strict
import os
import argparse
import json
import h5py
from math import ceil
from pathlib import Path
import numpy as np
import logging
from typing import Optional, Dict, List, Union, Any

import pynapple as nap
from one.api import ONE

from scipy.ndimage import gaussian_filter1d

from src.constants import SEED
from src.datasets.utils import *

from src.constants import DISCRETE_BEHAVIOR, CONTINUOUS_BEHAVIOR
from datasets import DatasetDict
from src.utils.dataset_utils import create_dataset

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and save binned spike data.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to base data directory")
    parser.add_argument("--eid", type=str, required=True, help="Experiment ID")
    parser.add_argument("--params", type=str, default=None,
                        help="Optional JSON string or path to a JSON file with parameter settings")
    parser.add_argument(
        "--streams", nargs="+", 
        default=["spikes", "choice", "block", "wheel", "left-whisker", "right-whisker"],
        help="List of data streams (e.g., spikes, behavior, lfp)"
    )
    parser.add_argument("--include_all_regions", 
        action="store_true", help="Include all brain regions in a session"
    )
    parser.add_argument(
        "--regions_of_interest", nargs="+", default=["CA1", "CA3", "DG", "LGd", "LP", "PO", "VISa"]
    )
    return parser.parse_args()

def load_params(params_arg: str | None) -> dict:
    default_params = {
        "interval_len": 1,
        "binsize": 0.0167,  # 60 Hz
        "single_region": False,
        "align_time": "stimOn_times",
        "time_window": (-0.2, 0.8),
        "behavior_keys": DISCRETE_BEHAVIOR + CONTINUOUS_BEHAVIOR,
        "fr_threshold": 1.0,  # 1 Hz
        "smoothing_sigma": 1.0, # 3 bins ~ 0.05s
    }
    if params_arg is None:
        return default_params
    try:
        # Try loading as JSON string
        return json.loads(params_arg)
    except json.JSONDecodeError:
        # Try loading as path to JSON file
        with open(params_arg, "r") as f:
            return json.load(f)

def filter_neurons_by_region(
    binned_spikes: np.ndarray, 
    meta_dict: Dict[str, Any],
    regions_of_interest: List[str],
) -> np.ndarray:
    
    cluster_regions = meta_dict["cluster_regions"]
    neurons_of_interest = np.isin(cluster_regions, regions_of_interest)

    # Filter spike data
    binned_spikes = binned_spikes[..., neurons_of_interest]

    # Filter metadata
    meta_dict["cluster_regions"] = list(
        np.array(cluster_regions, dtype=object)[neurons_of_interest]
    )
    meta_dict["cluster_channels"] = list(
        np.array(meta_dict["cluster_channels"], dtype=object)[neurons_of_interest]
    )
    meta_dict["uuids"] = list(
        np.array(meta_dict["uuids"], dtype=object)[neurons_of_interest]
    )
    return binned_spikes


def filter_active_neurons(
    binned_spikes: np.ndarray, meta_dict: Dict[str, Any], params: Dict[str, Any]
) -> np.ndarray:
    
    firing_rates = binned_spikes.mean(axis=(0,1)) / params["binsize"]  # (N,)
    neurons_of_interest = np.argwhere(firing_rates > params["fr_threshold"]).flatten()

    # Filter spike data
    binned_spikes = binned_spikes[..., neurons_of_interest]

    # Filter metadata
    meta_dict["cluster_regions"] = list(
        np.array(meta_dict["cluster_regions"], dtype=object)[neurons_of_interest]
    )
    meta_dict["cluster_channels"] = list(
        np.array(meta_dict["cluster_channels"], dtype=object)[neurons_of_interest]
    )
    meta_dict["uuids"] = list(
        np.array(meta_dict["uuids"], dtype=object)[neurons_of_interest]
    )
    return binned_spikes


def main(args):
    data_dir: Path = args.data_dir
    eid: str = args.eid
    params: Dict[str, Union[int, float, bool, str, tuple]] = load_params(args.params)
    streams: List[str] = args.streams

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=str(data_dir/"raw"),
    )#.setup()

    # Load data
    neural_dict, meta_dict, trials_data, behavior_data = prepare_data(one, eid, params)

    # Record dataset revisions for reproducibility
    # This is crucial for benchmark reproducibility as IBL datasets can have multiple revisions
    try:
        revisions = one.list_revisions(eid)
        if revisions:
            revision_info = ", ".join(revisions)
            meta_dict["dataset_revisions"] = revision_info
            logger.warning(f"DATASET REVISION WARNING: {eid} has multiple revisions: {revision_info}")
            logger.warning(f"This dataset revision info is saved to H5 metadata for reproducibility")
        else:
            meta_dict["dataset_revisions"] = "default"
            logger.info(f"Dataset {eid} uses default revision")
    except Exception as e:
        logger.warning(f"Could not retrieve revision info for {eid}: {e}")
        meta_dict["dataset_revisions"] = "unknown"

    regions, beryl_reg = list_brain_regions(meta_dict)
    logger.info(f"Available brain regions: {list(regions)}.")

    if args.include_all_regions:
        logger.info("Including all brain regions in the session.")
        regions = list(regions)
    else:
        regions = args.regions_of_interest
        logger.info(f"Filtering by brain regions of interest: {regions}.")

    # Get intervals
    trials_mask = trials_data["trials_mask"]
    trial_times = trials_data["trials_df"][params["align_time"]][trials_mask]
    start: np.ndarray = trial_times + params["time_window"][0]
    end: np.ndarray = trial_times + params["time_window"][1]
    intervals = nap.IntervalSet(start=start, end=end)

    # Construct spike timeseries
    cluster_ids: np.ndarray = np.unique(neural_dict["spike_clusters"])
    spikes = nap.TsGroup({
        cluster_id: nap.Ts(
            t=neural_dict["spike_times"][neural_dict["spike_clusters"] == cluster_id]
        ) for cluster_id in cluster_ids
    })

    # Bin and transpose
    binned_spikes = nap.build_tensor(
        spikes, intervals, bin_size=params["binsize"]
    ).transpose(1, 2, 0)
    logger.info(
        f"""Before region filtering:
            # Trials = {binned_spikes.shape[0]}
            # Time Bins = {binned_spikes.shape[1]}
            # Units = {binned_spikes.shape[2]}"""
    )

    # Filter by brain region if specified
    binned_spikes = filter_neurons_by_region(binned_spikes, meta_dict, regions_of_interest=regions)
    # Filter by firing rate
    binned_spikes = filter_active_neurons(binned_spikes, meta_dict, params)
    logger.info(
        f"""After region filtering:
            # Trials = {binned_spikes.shape[0]}
            # Time Bins = {binned_spikes.shape[1]}
            # Units = {binned_spikes.shape[2]}"""
    )

    # Smooth time bins using a Gaussian kernel
    smoothing_sigma = params.get("smoothing_sigma", 0)
    if smoothing_sigma > 0:
        logger.info(f"Smoothing spikes along time dimension with sigma={smoothing_sigma} bins.")
        binned_spikes = gaussian_filter1d(binned_spikes, sigma=smoothing_sigma, axis=1)

    # Bin behavior data
    behavior_dict = {}
    for behavior_key in params["behavior_keys"]:
        if behavior_key in DISCRETE_BEHAVIOR:
            behavior = behavior_data[behavior_key][trials_mask]
            unique_values = np.unique(behavior).tolist()
            logger.info(
                f"""Binning {behavior_key}:
                    # Trials = {behavior.shape[0]}
                    # Category = {len(unique_values)} with values {unique_values}"""
            )
        else:
            raw_behavior = nap.Tsd(
                t=behavior_data[behavior_key]["times"], 
                d=behavior_data[behavior_key]["values"],
            )
            num_bins = ceil(params["interval_len"] / params["binsize"])
            behavior = nap.warp_tensor(
                raw_behavior, intervals, num_bins=num_bins
            )
            logger.info(
                f"""Binning {behavior_key}:
                    # Trials = {behavior.shape[0]}
                    # Time Bins = {behavior.shape[1]}"""
            )
        behavior_dict[behavior_key] = behavior

    # Split by train / val / test
    rng = np.random.default_rng(SEED)
    perm_ids: np.ndarray = rng.permutation(len(intervals))
    # 70% for training, 10% for validation, 20% for testing
    train_boundary = int(len(intervals) * 0.7)
    val_boundary = train_boundary + int(len(intervals) * 0.1)
    partition_ids: Dict[str, np.ndarray] = {
        "train": perm_ids[:train_boundary],
        "val": perm_ids[train_boundary:val_boundary],
        "test": perm_ids[val_boundary:]
    }
    logger.info(
        f"Train: {len(partition_ids['train'])}\n"
        f"Val: {len(partition_ids['val'])}\n"
        f"Test: {len(partition_ids['test'])}"
    )

    # --- Save HuggingFace DatasetDict for MtM usage (train/val/test = 70/10/20) ---
    try:
        # create a single HF Dataset for all trials (create_dataset converts spikes to sparse lists and attaches per-sample meta)
        ds_all = create_dataset(binned_spikes, None, eid, params, meta_data=meta_dict, binned_behaviors=behavior_dict)

        # select splits using partition_ids computed above
        ds_train = ds_all.select(partition_ids['train'].tolist())
        ds_val = ds_all.select(partition_ids['val'].tolist())
        ds_test = ds_all.select(partition_ids['test'].tolist())

        ds_dict = DatasetDict({
            "train": ds_train,
            "val": ds_val,
            "test": ds_test
        })

        # save HF DatasetDict to disk (aligned dir)
        out_dir = data_dir / f"{eid}_aligned"
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_dict.save_to_disk(str(out_dir))

        # save session-level metadata json for convenience
        meta_out = out_dir / "session_metadata.json"
        meta_to_save = {
            "eid": meta_dict.get("eid", eid),
            "subject": meta_dict.get("subject"),
            "lab": meta_dict.get("lab"),
            "cluster_regions": meta_dict.get("cluster_regions"),
            "cluster_channels": meta_dict.get("cluster_channels"),
            "uuids": meta_dict.get("uuids"),
            "sampling_freq": meta_dict.get("sampling_freq", None),
            "binsize": params.get("binsize"),
            "interval_len": params.get("interval_len"),
            "dataset_revisions": meta_dict.get("dataset_revisions", None),
        }
        with open(meta_out, "w") as fh:
            json.dump(meta_to_save, fh, indent=2)

        logger.info(f"Saved HuggingFace DatasetDict to {out_dir} (train/val/test sizes: {len(ds_train)}/{len(ds_val)}/{len(ds_test)})")
    except Exception as e:
        logger.warning(f"Could not save HuggingFace DatasetDict: {e}")

    # Save to HDF5
    for partition, ids in partition_ids.items():
        output_path: Path = data_dir / "precached" / f"{partition}_{eid}.h5"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if partition == "train":
            trial_avg_metadata = {}
            trial_avg_spikes = np.mean(binned_spikes, axis=0)
            # binned_spikes = binned_spikes - trial_avg_spikes
            trial_avg_metadata["trial_avg_spikes"] = trial_avg_spikes
            for behavior_key in params["behavior_keys"]:
                if behavior_key in CONTINUOUS_BEHAVIOR:
                    trial_avg_behavior = np.mean(behavior_dict[behavior_key], axis=0)
                    # behavior_dict[behavior_key] = behavior_dict[behavior_key] - trial_avg_behavior
                    trial_avg_metadata[f"trial_avg_{behavior_key}"] = trial_avg_behavior

        with h5py.File(output_path, "w") as f:
            for stream in streams:
                if stream == "spikes":
                    f.create_dataset("spikes", data=binned_spikes[ids])
                elif stream in params["behavior_keys"]:
                    f.create_dataset(stream, data=behavior_dict[stream][ids])
                else:
                    # TODO Handle other streams if needed
                    raise NotImplementedError(f"Stream '{stream}' is not implemented.")
            f.create_dataset("start", data=intervals.start[ids])
            f.create_dataset("end", data=intervals.end[ids])      

            for k, v in trial_avg_metadata.items():
                f.create_dataset(k, data=v)      

            meta_grp = f.create_group("metadata")
            base_dir = os.path.dirname(os.path.abspath(__file__))
            for k, v in meta_dict.items():
                meta_grp.attrs[k] = v
                if k == "uuids":
                    # Save UUIDs to a local text file
                    uuid_path = os.path.join(base_dir, "uuids.txt")
                    with open(uuid_path, "w") as f:
                        for uuid in v:
                            f.write(str(uuid) + "\n")

            # Save brain regions to a local text file
            unique_regions = np.unique(meta_dict["cluster_regions"]).tolist()
            regions_path = os.path.join(base_dir, "regions.txt")
            with open(regions_path, "w") as f:
                for region in unique_regions:
                    f.write(str(region) + "\n")

        logger.info(f"Precached data saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)