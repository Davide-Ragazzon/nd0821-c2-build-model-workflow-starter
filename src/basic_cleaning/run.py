#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def perform_simple_cleaning(df, min_price=10, max_price=350):
    # Drop outliers
    idx = df["price"].between(min_price, max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])
    return df


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_local_path = artifact.file()

    ori_df = pd.read_csv(artifact_local_path)
    clean_df = perform_simple_cleaning(ori_df, min_price=args.min_price, max_price=args.max_price)

    output_file = "clean_sample.csv"
    clean_df.to_csv(output_file, index=False)
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", type=str, help="Input artifact with the raw data", required=True
    )

    parser.add_argument(
        "--output_artifact", type=str, help="Output artifact with the cleaned data", required=True
    )

    parser.add_argument("--output_type", type=str, help="Output type", required=True)

    parser.add_argument("--output_description", type=str, help="Output description", required=True)

    parser.add_argument(
        "--min_price", type=float, help="Minumum price for outlier exclusion", required=True
    )

    parser.add_argument(
        "--max_price", type=float, help="Maximum price for outlier exclusion", required=True
    )

    args = parser.parse_args()

    go(args)
