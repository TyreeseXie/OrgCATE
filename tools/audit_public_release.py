"""Audit a candidate OrgCATE tree before it is staged or uploaded."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "PUBLIC_RELEASE_MANIFEST.json"

IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "local_outputs",
    "results",
    "venv",
    "AGENTS.md",
}
FORBIDDEN_PARTS = {
    "access_notes",
    "private",
    "processed",
    "raw",
}
FORBIDDEN_SUFFIXES = {
    ".7z",
    ".bak",
    ".doc",
    ".docx",
    ".dta",
    ".gz",
    ".joblib",
    ".log",
    ".parquet",
    ".pkl",
    ".ppt",
    ".pptx",
    ".sas7bdat",
    ".sav",
    ".tar",
    ".tmp",
    ".xls",
    ".xlsx",
    ".zip",
}
ALLOWED_TABULAR = {
    "execution/phase2_real_backbone_v1/anchor_selection_distribution_L3.csv",
    "execution/phase2_real_backbone_v1/backbone_feature_dictionary_v1.csv",
    "execution/phase2_real_backbone_v1/candidate_feature_pearson_correlation_L3.csv",
    "execution/phase2_real_backbone_v1/candidate_feature_spearman_correlation_L3.csv",
    "execution/phase2_real_backbone_v1/feature_support_qc_v1.csv",
    "execution/phase2_real_backbone_v1/history_length_sample_composition.csv",
    "reports/kink_resolution_v2/KINK_RESOLUTION_V2_DEVELOPMENT_LONG.csv",
    "reports/kink_resolution_v2/KINK_RESOLUTION_V2_SUMMARY.csv",
    "reports/morphology_product_v1/MORPHOLOGY_PRODUCT_V1_DEVELOPMENT_LONG.csv",
    "reports/morphology_product_v1/MORPHOLOGY_PRODUCT_V1_SUMMARY.csv",
    "reports/nonlinear_monotone_resolution_v1/confirmation/NONLINEAR_CONFIRMATION_EVALUATOR_LONG.csv",
    "reports/nonlinear_monotone_resolution_v1/confirmation/NONLINEAR_CONFIRMATION_PRACTICAL_LONG.csv",
    "reports/nonlinear_monotone_resolution_v1/confirmation/NONLINEAR_CONFIRMATION_SUMMARY.csv",
    "reports/single_scenario_release_candidate_v1/SINGLE_SCENARIO_UNIFIED_SUMMARY.csv",
    "reports/unified_recovery_acceptance_v1/UNIFIED_SINGLE_SUMMARY.csv",
}
ALLOWED_BINARY_DATA = {
    "execution/phase2_real_backbone_v1/headline_one_anchor_per_firm_L3.pkl.gz": {
        "bytes": 2643021,
        "sha256": "91c963e02cbdc17a19de85868ba44f5f4fd65791bec83906980f307a6cfd3174",
    }
}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".gitignore",
    ".json",
    ".md",
    ".mmd",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ABSOLUTE_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\")
ABSOLUTE_UNIX_HOME = re.compile(r"/(?:Users|home)/[^/\s]+/")
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[\"']?([^\s\"']+)"
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
PLACEHOLDER_VALUES = {
    "[confirm]",
    "[choose",
    "[add",
    "none",
    "not_set",
    "pending",
    "replace_me",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_bytes(path: Path) -> bytes:
    """Return the bytes Git records for cross-platform manifest hashes."""

    content = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        return content.replace(b"\r\n", b"\n")
    return content


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def candidate_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if path.is_dir() or path == MANIFEST_PATH:
            continue
        parts = path.relative_to(ROOT).parts
        if any(part in IGNORED_PARTS for part in parts):
            continue
        if any(part.startswith((".pycache", ".pytest")) for part in parts):
            continue
        files.append(path)
    return sorted(files, key=relative)


def inspect_csv(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            return 0, []
        rows = sum(1 for _ in reader)
    return rows, header


def inspect_text(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if path.name == Path(__file__).name:
        return findings
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        findings.append(
            {"severity": "ERROR", "path": relative(path), "issue": "non-UTF-8 text"}
        )
        return findings
    if ABSOLUTE_WINDOWS_PATH.search(text) or ABSOLUTE_UNIX_HOME.search(text):
        findings.append(
            {
                "severity": "ERROR",
                "path": relative(path),
                "issue": "machine-specific absolute path",
            }
        )
    for match in SECRET_ASSIGNMENT.finditer(text):
        value = match.group(1).strip().lower()
        if not any(value.startswith(item) for item in PLACEHOLDER_VALUES):
            findings.append(
                {
                    "severity": "ERROR",
                    "path": relative(path),
                    "issue": "possible credential assignment",
                }
            )
            break
    if path.suffix.lower() == ".md":
        for raw_target in MARKDOWN_LINK.findall(text):
            target = unquote(raw_target.strip().strip("<>")).split("#", 1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            linked = (path.parent / target).resolve()
            try:
                linked.relative_to(ROOT)
            except ValueError:
                findings.append(
                    {
                        "severity": "ERROR",
                        "path": relative(path),
                        "issue": f"local Markdown link escapes repository: {target}",
                    }
                )
                continue
            if not linked.exists():
                findings.append(
                    {
                        "severity": "ERROR",
                        "path": relative(path),
                        "issue": f"broken local Markdown link: {target}",
                    }
                )
    return findings


def audit(*, allow_metadata_pending: bool) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    files = candidate_files()
    for path in files:
        rel = relative(path)
        parts = {part.lower() for part in path.relative_to(ROOT).parts}
        suffix = path.suffix.lower()
        if path.is_symlink():
            findings.append(
                {"severity": "ERROR", "path": rel, "issue": "symbolic link"}
            )
        if parts.intersection(FORBIDDEN_PARTS):
            findings.append(
                {"severity": "ERROR", "path": rel, "issue": "private-data path"}
            )
        if suffix in FORBIDDEN_SUFFIXES and rel not in ALLOWED_BINARY_DATA:
            findings.append(
                {"severity": "ERROR", "path": rel, "issue": "forbidden file type"}
            )
        if rel in ALLOWED_BINARY_DATA:
            expected = ALLOWED_BINARY_DATA[rel]
            observed_hash = sha256(path)
            if path.stat().st_size != expected["bytes"] or observed_hash != expected["sha256"]:
                findings.append(
                    {
                        "severity": "ERROR",
                        "path": rel,
                        "issue": "authorized backbone size/hash mismatch",
                    }
                )
        if suffix == ".csv":
            rows, header = inspect_csv(path)
            if rel not in ALLOWED_TABULAR:
                findings.append(
                    {
                        "severity": "ERROR",
                        "path": rel,
                        "issue": "tabular file is not on aggregate-evidence allowlist",
                    }
                )
            if rows > 100:
                findings.append(
                    {
                        "severity": "ERROR",
                        "path": rel,
                        "issue": f"aggregate table unexpectedly has {rows} rows",
                    }
                )
            lowered = {name.strip().lower() for name in header}
            if lowered.intersection({"gvkey", "unit_id", "firm_id", "company_id"}):
                findings.append(
                    {
                        "severity": "ERROR",
                        "path": rel,
                        "issue": "possible row-level firm identifier column",
                    }
                )
        if suffix in TEXT_SUFFIXES:
            findings.extend(inspect_text(path))
        content = manifest_bytes(path)
        records.append(
            {
                "path": rel,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )

    metadata_missing = [
        name for name in ("LICENSE", "CITATION.cff") if not (ROOT / name).is_file()
    ]
    if metadata_missing and not allow_metadata_pending:
        findings.append(
            {
                "severity": "BLOCKER",
                "path": ".",
                "issue": "missing required publication metadata: "
                + ", ".join(metadata_missing),
            }
        )
    status = "PASS" if not findings else "BLOCKED"
    return {
        "schema_version": "orgcate-public-release-audit-v2",
        "status": status,
        "allow_metadata_pending": allow_metadata_pending,
        "metadata_missing": metadata_missing,
        "manifest_excludes_itself": True,
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "findings": findings,
        "files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-metadata-pending",
        action="store_true",
        help="permit absent LICENSE/CITATION.cff while building the candidate",
    )
    args = parser.parse_args()
    report = audit(allow_metadata_pending=args.allow_metadata_pending)
    MANIFEST_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{report['status']}: {report['file_count']} files, "
        f"{len(report['findings'])} finding(s)"
    )
    for item in report["findings"]:
        print(f"{item['severity']}  {item['path']}: {item['issue']}")
    if report["metadata_missing"]:
        print("PENDING  " + ", ".join(report["metadata_missing"]))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())

