"""Static pair ownership and exact report merging."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic_multiturn_evals.campaign import (
    CampaignManifest,
    Report,
    _exclusive_lock,
    _exit_code,
    _manifest,
    _result_order,
    _trials,
    read_report,
)
from pydantic_multiturn_evals.model import Campaign, TrialKey, TrialResult


def merge_reports(campaign: Campaign, roots: Sequence[Path]) -> Report:
    """Merge one closed root per shard and reject every gap or duplicate."""

    if not roots:
        raise ValueError("at least one shard root is required")
    loaded = [_read_closed_shard(root.resolve()) for root in roots]
    first_manifest = loaded[0][0]
    if first_manifest.shard is None:
        raise ValueError("report root is not a shard")
    shard_count = first_manifest.shard.count
    if len(loaded) != shard_count:
        raise ValueError(f"expected {shard_count} shard roots, got {len(loaded)}")

    by_index: dict[int, tuple[CampaignManifest, Report]] = {}
    common = first_manifest.model_dump(mode="json", exclude={"shard"})
    for manifest, report in loaded:
        shard = manifest.shard
        if shard is None or shard.count != shard_count:
            raise ValueError("shard roots have incompatible shard declarations")
        if manifest.model_dump(mode="json", exclude={"shard"}) != common:
            raise ValueError("shard roots contain different campaign definitions")
        expected_manifest = _manifest(campaign, manifest.artifact_limits, shard)
        if manifest != expected_manifest:
            raise ValueError(f"shard {shard.index} does not match the supplied campaign")
        if shard.index in by_index:
            raise ValueError(f"shard index {shard.index} is present more than once")
        by_index[shard.index] = (manifest, report)
    missing_shards = set(range(shard_count)) - by_index.keys()
    if missing_shards:
        values = ", ".join(str(index) for index in sorted(missing_shards))
        raise ValueError(f"missing shard indexes: {values}")

    results: list[TrialResult] = []
    seen: set[tuple[str, str, str, str, int]] = set()
    for index in range(shard_count):
        manifest, report = by_index[index]
        shard = manifest.shard
        assert shard is not None
        expected = {
            _key_identity(key)
            for key, _case, _variant in _trials(campaign, shard)
        }
        actual = {_key_identity(result.key) for result in report.results}
        if len(actual) != len(report.results):
            raise ValueError(f"shard {index} contains duplicate trial results")
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            detail = _key_text(next(iter(missing or extra)))
            problem = "missing" if missing else "unexpected"
            raise ValueError(f"shard {index} has {problem} trial result {detail}")
        duplicate = actual & seen
        if duplicate:
            key = _key_text(next(iter(duplicate)))
            raise ValueError(f"trial result is present in multiple shards: {key}")
        seen.update(actual)
        results.extend(report.results)

    ordered = tuple(sorted(results, key=_result_order))
    return Report(
        campaign=campaign.name,
        revision=campaign.revision,
        executed_trials=0,
        results=ordered,
        exit_code=_exit_code(ordered),
    )


def _read_closed_shard(root: Path) -> tuple[CampaignManifest, Report]:
    with _exclusive_lock(root / ".lock"):
        manifest = CampaignManifest.model_validate_json(
            (root / "campaign.json").read_text(encoding="utf-8")
        )
        return manifest, read_report(root)


def _key_identity(key: TrialKey) -> tuple[str, str, str, str, int]:
    return key.campaign, key.revision, key.case, key.variant, key.sample


def _key_text(key: tuple[str, str, str, str, int]) -> str:
    campaign, revision, case, variant, sample = key
    return f"{campaign}/{revision}/{case}/{variant}/sample-{sample}"
