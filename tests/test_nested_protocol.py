import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.nested_folds import generate_nested_folds
from src.evaluation.evaluate_heldout_test import evaluate
from src.evaluation.outer_test_sweep import descriptive_sweep
from src.evaluation.select_operating_point import THRESHOLDS, select_threshold
from src.statistics.summarize_folds import summarize


def prediction_row(model="m", outer_fold=0, inner_fold=0, split="inner_val", score=0.6):
    return {
        "image_path": f"image_{inner_fold}.png",
        "patient_id": f"p{inner_fold}",
        "model": model,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "split": split,
        "gt_boxes": json.dumps([[0, 0, 10, 10]]),
        "pred_boxes": json.dumps([[0, 0, 10, 10]]),
        "pred_scores": json.dumps([score]),
    }


def test_nested_folds_are_patient_level_and_complete():
    manifest = pd.DataFrame(
        [
            {"image_path": f"i{patient}_{image}.png", "label_path": "x.txt", "patient_id": str(patient)}
            for patient in range(20)
            for image in range(2)
        ]
    )
    outputs = generate_nested_folds(manifest, seed=7)
    assert len(outputs) == 5
    outer_test_sets = []
    for outer_fold, fold in enumerate(outputs):
        assert set(fold.outer_fold) == {outer_fold}
        patient_splits = fold.groupby("patient_id").split.nunique()
        assert patient_splits.max() == 1
        test = set(fold.loc[fold.split == "outer_test", "patient_id"])
        train = set(fold.loc[fold.split == "outer_train", "patient_id"])
        assert not test & train
        outer_test_sets.append(test)
        assignments = fold.loc[fold.split == "outer_train"].groupby("patient_id").inner_fold.nunique()
        assert assignments.eq(1).all()
        assert set(fold.loc[fold.split == "outer_train", "inner_fold"].astype(int)) == {0, 1, 2, 3}
    assert set.union(*outer_test_sets) == set(manifest.patient_id)
    assert sum(len(group) for group in outer_test_sets) == manifest.patient_id.nunique()


def test_threshold_grid_mean_inner_f1_and_high_tie(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,label_path,patient_id\ni,l,p\n", encoding="utf-8")
    paths = []
    for inner_fold, score in enumerate([0.6, 0.6, 0.8, 0.8]):
        path = tmp_path / f"inner_{inner_fold}.csv"
        pd.DataFrame([prediction_row(inner_fold=inner_fold, score=score)]).to_csv(path, index=False)
        paths.append(path)
    payload, curves = select_threshold(paths, "m", 0, manifest)
    assert len(THRESHOLDS) == 101
    assert curves.threshold.nunique() == 101
    assert len(curves) == 404
    assert payload["confidence_threshold"] == pytest.approx(0.6)
    assert payload["selected_mean_f1"] == pytest.approx(1.0)
    assert [item["inner_fold"] for item in payload["inner_folds"]] == [0, 1, 2, 3]
    assert payload["manifest"]["sha256"]


def test_test_evaluation_rejects_wrong_identity_and_applies_locked_threshold():
    locked = {"model": "m", "outer_fold": 0, "confidence_threshold": 0.7, "matching_iou": 0.5}
    row = prediction_row(split="outer_test", score=0.6)
    df = pd.DataFrame([row])
    _, _, summary = evaluate(df, locked)
    assert summary["threshold"] == 0.7
    assert summary["tp"] == 0 and summary["fn"] == 1
    with pytest.raises(ValueError, match="model"):
        evaluate(df, {**locked, "model": "other"})
    with pytest.raises(ValueError, match="outer_fold"):
        evaluate(df, {**locked, "outer_fold": 1})


def write_summary(path, fold, split="outer_test", model="m"):
    path.write_text(
        json.dumps(
            {
                "model": model,
                "outer_fold": fold,
                "evaluation_split": split,
                "precision": fold / 4,
                "sensitivity": 1.0,
                "f1": 0.5,
                "fp_per_image": 0.1,
            }
        ),
        encoding="utf-8",
    )


def test_strict_five_fold_summary_constraints(tmp_path):
    paths = []
    for fold in range(5):
        path = tmp_path / f"fold_{fold}.json"
        write_summary(path, fold)
        paths.append(path)
    result = summarize(paths)
    precision = result[result.metric == "precision"].iloc[0]
    assert precision["mean"] == pytest.approx(0.5)
    assert precision["sample_sd"] == pytest.approx(pd.Series([0, .25, .5, .75, 1]).std(ddof=1))
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(paths[:4] + [paths[3]])
    bad = tmp_path / "bad.json"
    write_summary(bad, 4, split="inner_val")
    with pytest.raises(ValueError, match="outer_test"):
        summarize(paths[:4] + [bad])


def test_descriptive_outer_test_sweep_does_not_change_locked_threshold():
    locked = {"model": "m", "outer_fold": 0, "confidence_threshold": 0.7, "matching_iou": 0.5}
    before = locked.copy()
    curves = descriptive_sweep(pd.DataFrame([prediction_row(split="outer_test")]), locked)
    assert len(curves) == 101
    assert curves.descriptive_only.all()
    assert curves.locked_threshold.eq(0.7).all()
    assert locked == before
