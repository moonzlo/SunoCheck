"""Factory (−1 st) LR path (research H29)."""

from pathlib import Path

import numpy as np
import soundfile as sf

from sunocheck.factory_fakeprint import (
    FACTORY_MODEL_PATH,
    FactoryScore,
    factory_joint,
    factory_score,
    load_factory_model,
)

RECORDINGS: Path = Path(__file__).resolve().parents[1] / "data" / "recordings"


def test_factory_model_loads() -> None:
    model = load_factory_model(FACTORY_MODEL_PATH)
    assert model.weights.size == model.bins.size
    assert model.weights.size > 1000


def test_factory_joint_rule() -> None:
    assert factory_joint(FactoryScore(score=0.0, lock_contrast=2.0, ratio=0.9438))
    assert not factory_joint(FactoryScore(score=-0.1, lock_contrast=5.0, ratio=0.9438))
    assert not factory_joint(FactoryScore(score=1.0, lock_contrast=1.9, ratio=0.9438))
    assert not factory_joint(None)


def test_live_velkira_joint_fires() -> None:
    matches = sorted(RECORDINGS.glob("*Velkira*"))
    if not matches:
        return
    model = load_factory_model(FACTORY_MODEL_PATH)
    audio, rate = sf.read(matches[0], dtype="float32", always_2d=True)
    score = factory_score(audio.mean(axis=1).astype(np.float32), rate, model)
    assert factory_joint(score)


def test_live_human_score_does_not_joint() -> None:
    matches = sorted(RECORDINGS.glob("*Lyra*Falling*"))
    if not matches:
        matches = sorted(RECORDINGS.glob("*Thirty Seconds*"))
    if not matches:
        return
    model = load_factory_model(FACTORY_MODEL_PATH)
    audio, rate = sf.read(matches[0], dtype="float32", always_2d=True)
    score = factory_score(audio.mean(axis=1).astype(np.float32), rate, model)
    assert not factory_joint(score)


def test_live_ravnir_joint_via_slide() -> None:
    """First-loud window alone misses; sliding best (H29 variants) recovers RAVNIR."""
    matches = sorted(RECORDINGS.glob("*RAVNIR*M*rk*"))
    if not matches:
        matches = sorted(RECORDINGS.glob("*RAVNIR*"))
    if not matches:
        return
    model = load_factory_model(FACTORY_MODEL_PATH)
    audio, rate = sf.read(matches[0], dtype="float32", always_2d=True)
    score = factory_score(audio.mean(axis=1).astype(np.float32), rate, model)
    assert factory_joint(score)


def test_live_dust_still_below_joint() -> None:
    """Soft joint rejected on corpus FPR; Dust remains a known miss at stock thresholds."""
    matches = sorted(RECORDINGS.glob("*Dust*Wolves*"))
    if not matches:
        return
    model = load_factory_model(FACTORY_MODEL_PATH)
    audio, rate = sf.read(matches[0], dtype="float32", always_2d=True)
    score = factory_score(audio.mean(axis=1).astype(np.float32), rate, model)
    assert not factory_joint(score)
