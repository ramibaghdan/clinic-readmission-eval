"""Integrity tests: the guardrails that keep the two traps from creeping back.

These run against the cached dataset (downloaded by `python data_prep.py` or
`make data`). They are fast and assert the properties the finding depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import data_prep as dp


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return dp.load_raw()


# --------------------------------------------------------------------------- #
# ICD-9 grouping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "code,expected",
    [
        ("250.83", "Diabetes"),
        ("428", "Circulatory"),
        ("786", "Respiratory"),
        ("V57", "Other"),
        ("E909", "Other"),
        ("800", "Injury"),
        ("715", "Musculoskeletal"),
        ("185", "Neoplasms"),
        ("?", "Missing"),
        (np.nan, "Missing"),
    ],
)
def test_icd9_grouping(code, expected):
    assert dp.map_icd9_group(code) == expected


# --------------------------------------------------------------------------- #
# Issue 2 -- cohort exclusion
# --------------------------------------------------------------------------- #
def test_naive_cohort_keeps_all_rows(raw):
    coh, report = dp.build_cohort(raw, exclude_expired_hospice=False)
    assert report["n_excluded_expired_hospice"] == 0
    assert len(coh) == len(raw)


def test_corrected_cohort_drops_expired_hospice(raw):
    coh, report = dp.build_cohort(raw, exclude_expired_hospice=True)
    assert report["n_excluded_expired_hospice"] > 0
    remaining = set(coh["discharge_disposition_id"].unique())
    assert remaining.isdisjoint(set(dp.EXPIRED_HOSPICE_DISPOSITION_IDS))


# --------------------------------------------------------------------------- #
# Issue 3 -- binarization
# --------------------------------------------------------------------------- #
def test_binarization_is_lt30_only(raw):
    coh = dp.binarize_target(raw)
    pos = coh.loc[coh[dp.TARGET] == 1, dp.TARGET_RAW].unique()
    assert set(pos) == {"<30"}


# --------------------------------------------------------------------------- #
# Issue 1 -- split integrity
# --------------------------------------------------------------------------- #
def test_grouped_split_has_zero_patient_overlap(raw):
    coh = dp.binarize_target(raw)
    tr, te = dp.grouped_split(coh[dp.TARGET], coh[dp.PATIENT_ID])
    overlap = dp.patient_overlap(coh[dp.PATIENT_ID], tr, te)
    assert overlap["n_overlapping_patients"] == 0


def test_stratified_split_leaks_patients(raw):
    """The naive split is *supposed* to leak -- that's the point we measure."""
    coh = dp.binarize_target(raw)
    tr, te = dp.stratified_split(coh[dp.TARGET])
    overlap = dp.patient_overlap(coh[dp.PATIENT_ID], tr, te)
    assert overlap["n_overlapping_patients"] > 0


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def test_zero_variance_columns_dropped(raw):
    coh = dp.binarize_target(raw)
    X, cat_features, meta = dp.build_features(coh)
    for col in ("examide", "citoglipton", "weight"):
        assert col not in X.columns


def test_no_missing_in_categoricals(raw):
    coh = dp.binarize_target(raw)
    X, cat_features, meta = dp.build_features(coh)
    for c in cat_features:
        assert X[c].isna().sum() == 0
