#!/usr/bin/env python3
"""
Cirro preprocess script for ebbf pipeline.

This script prepares the samplesheet.csv input file required by the
nf-core/ebbf Nextflow pipeline.

The pipeline accepts either:
- SpatialData zarr directories
- Xenium Ranger output directories

The format is auto-detected by the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from cirro.helpers.preprocess_dataset import PreprocessDataset

SAMPLESHEET_REQUIRED_COLUMNS = (
    "sample",
    "zarr_dir",
)


def samplesheet_from_files(ds: PreprocessDataset) -> pd.DataFrame:
    """
    Create a samplesheet from Cirro's files DataFrame.

    Uses ds.files to get files, creates zarr_dir from parent of each file,
    and merges with ds.samplesheet.
    """
    files = ds.files

    ds.logger.info(f"Found files in ds.files: {files.to_dict()}")

    # Convert s3 link to PosixPath and derive parent; convert back into string
    # Path converts s3:// to s3:/, so revert proper s3 prefix afterwards
    files = files.copy()
    files["zarr_dir"] = files["file"].apply(lambda x: str(Path(x).parent).replace("s3:/", "s3://"))
    files = files[["sample", "zarr_dir"]]

    # Deduplicate - keep one row per sample
    # (all files for a sample share same zarr_dir)
    files = files.drop_duplicates(subset=["sample"])

    samplesheet = pd.merge(ds.samplesheet, files, on="sample", how="left")

    return samplesheet


def samplesheet_from_params(ds: PreprocessDataset) -> pd.DataFrame:
    """
    Create a samplesheet from dataset metadata inputs.

    Falls back to using ds.metadata['inputs'] when no files are found.
    """
    data_params = pd.DataFrame(
        {
            "sample": [x["name"] for x in ds.metadata["inputs"]],
            "zarr_dir": [x["dataPath"] for x in ds.metadata["inputs"]],
        }
    )

    return data_params


def prepare_samplesheet(ds: PreprocessDataset) -> pd.DataFrame:
    """
    Prepare the samplesheet for the pipeline.

    Tries to create from files first, falls back to params if no files found.
    Ensures all required columns are present and cleans up params.
    """
    ds.logger.info(f"Params: {ds.params}")

    samplesheet = samplesheet_from_files(ds)

    # Check if pipeline uses Cirro samplesheet, and if not prepare it from params
    if samplesheet.empty:
        ds.logger.warning("No files found in dataset. Preparing samplesheet from params.")
        samplesheet = samplesheet_from_params(ds)
        if samplesheet.empty:
            raise ValueError(
                "No files found in dataset and unable to prepare samplesheet from params."
            )
        ds.logger.info("Prepared samplesheet from params.")

    # Ensure all required columns are present (populate missing)
    for colname in SAMPLESHEET_REQUIRED_COLUMNS:
        if colname not in samplesheet.columns:
            ds.logger.warning(
                f"Samplesheet is missing required column '{colname}'. Populating with NaN."
            )
            samplesheet[colname] = np.nan

    # Save to a file
    samplesheet.to_csv("samplesheet.csv", index=False)

    # Clear params that we wrote to the samplesheet
    # cleared params will not overload the nextflow.params
    to_remove = []
    for k in ds.params:
        if k in SAMPLESHEET_REQUIRED_COLUMNS:
            ds.logger.info(
                f"Removing param '{k}' from dataset params as it is now in the samplesheet."
            )
            to_remove.append(k)

    for k in to_remove:
        ds.remove_param(k)

    ds.add_param("input", "samplesheet.csv", overwrite=True)

    # Log the samplesheet
    ds.logger.info(f"Samplesheet: {samplesheet.to_dict()}")

    return samplesheet


def main():
    ds = PreprocessDataset.from_running()

    ds.logger.info("Creating samplesheet from input files")
    ds.logger.info(f"Input files: {len(ds.files)} rows")
    ds.logger.info(f"Input columns: {list(ds.files.columns)}")
    ds.logger.info(f"Files DataFrame:\n{ds.files.to_string()}")

    samplesheet = prepare_samplesheet(ds)
    ds.logger.info(f"Samplesheet created with {len(samplesheet)} samples")

    # Log final params
    ds.logger.info(f"Final params: {ds.params}")


if __name__ == "__main__":
    main()
