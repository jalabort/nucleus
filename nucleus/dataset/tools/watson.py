from typing import Optional, Union, Iterable, Dict

import re
import json
import boto3
import pandas as pd

from hudl_aws.s3 import read_from_s3

from nucleus.types import Num
from nucleus.utils import progress_bar

from ..keys import DatasetKeys

from .shared import create_df_from_examples


__all__ = [
    'get_job_keys',
    'get_jobs',
    'create_examples_from_jobs',
    'create_df_from_s3'
]


def get_job_keys(bucket: str, key: str, pattern: str) -> Iterable[str]:
    r"""

    Parameters
    ----------
    bucket
    key
    pattern

    Returns
    -------

    """
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects')

    regex = re.compile(key + pattern)

    for result in paginator.paginate(Bucket=bucket, Prefix=key):
        for contents in result.get('Contents'):
            key_path = contents.get('Key')
            match = re.search(regex, key_path)
            if match is not None:
                yield key_path


def get_jobs(
        bucket: str,
        key: str,
        pattern: str,
        n_jobs: Optional[int] = None,
        show_progress: bool = False
) -> Iterable[Dict[str, Iterable]]:
    r"""

    Parameters
    ----------
    bucket
    key
    pattern
    n_jobs
    show_progress

    Returns
    -------

    """
    keys = get_job_keys(bucket=bucket, key=key, pattern=pattern)

    if show_progress:
        keys = progress_bar(list(keys)[:n_jobs])

    for key in keys:
        yield json.load(read_from_s3(bucket, key))


def create_examples_from_jobs(
        jobs: Iterable[Dict[str, Iterable]]
) -> Iterable[Dict[str, Union[Num, str]]]:
    r"""

    Parameters
    ----------
    jobs

    Returns
    -------

    """
    for job in jobs:
        # TODO: Should examples be called `images`
        for example in job['examples']:
            ijhw_list = []
            labels_list = []
            for parsed in example['boxes']:
                # TODO: y and x here represent j and i
                ijhw = [
                    parsed['y'],
                    parsed['x'],
                    parsed['height'],
                    parsed['width']
                ]
                labels = [parsed['label']]
                ijhw_list.append(ijhw)
                labels_list.append(labels)

            yield {
                DatasetKeys.PATH.value: example['source'],
                DatasetKeys.LABELS.value: example['frameTags'],
                DatasetKeys.BOXES.value: ijhw_list,
                DatasetKeys.BOXES_LABELS.value: labels_list,
                DatasetKeys.N_BOXES.value: len(ijhw_list),
                'src': 'watson'
            }


def create_df_from_s3(
        bucket: str,
        key: str,
        pattern: str,
        n_jobs: Optional[int] = None,
        show_progress: bool = False
) -> pd.DataFrame:
    r"""

    Parameters
    ----------
    bucket
    key
    pattern
    n_jobs
    show_progress

    Returns
    -------

    """
    jobs = get_jobs(
        bucket=bucket,
        key=key,
        pattern=pattern,
        n_jobs=n_jobs,
        show_progress=show_progress
    )
    examples = create_examples_from_jobs(jobs)
    df = create_df_from_examples(examples)
    return df
