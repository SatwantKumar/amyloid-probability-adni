#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adni_analysis.citations import (  # noqa: E402
    PubMedError,
    extract_cite_keys,
    format_ama_journal_reference,
    format_nature_journal_reference,
    pubmed_fetch_articles,
    pubmed_search_pmids,
    replace_cite_tags,
    resolve_doi_to_pmid,
)


def _format_ref(article: Any, *, style: str) -> str:
    if style == "ama":
        return format_ama_journal_reference(article)
    if style == "nature":
        return format_nature_journal_reference(article)
    raise ValueError(f"Unknown reference style: {style}")


def _load_refs_yaml(path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("References YAML must be a mapping")
    citations = payload.get("citations", {})
    if not isinstance(citations, dict):
        raise ValueError("References YAML must contain a 'citations:' mapping")
    out: dict[str, dict[str, Any]] = {}
    for key, spec in citations.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid citation spec for {key}: expected mapping")
        out[str(key)] = dict(spec)
    return out


def _resolve_specs_to_pmids(
    key_to_spec: dict[str, dict[str, Any]],
    *,
    email: str | None,
    tool: str | None,
    api_key: str | None,
) -> dict[str, str]:
    key_to_pmid: dict[str, str] = {}
    for key, spec in key_to_spec.items():
        pmid = spec.get("pmid")
        doi = spec.get("doi")
        if pmid and doi:
            raise ValueError(f"Citation {key} specifies both pmid and doi; choose one.")
        if pmid:
            key_to_pmid[key] = str(pmid).strip()
            continue
        if doi:
            key_to_pmid[key] = resolve_doi_to_pmid(
                str(doi),
                tool=tool,
                email=email,
                api_key=api_key,
            )
            continue
        raise ValueError(f"Citation {key} must specify either pmid or doi.")
    return key_to_pmid


def cmd_search(args: argparse.Namespace) -> None:
    pmids = pubmed_search_pmids(
        args.query,
        retmax=args.retmax,
        tool=args.tool,
        email=args.email,
        api_key=args.api_key,
    )
    if not pmids:
        print("No results.")
        return
    articles = pubmed_fetch_articles(pmids, tool=args.tool, email=args.email, api_key=args.api_key)
    for a in articles:
        line = _format_ref(a, style=args.style)
        print(f"{a.pmid}\t{line}")


def cmd_format(args: argparse.Namespace) -> None:
    pmids = [p.strip() for p in args.pmids.split(",") if p.strip()]
    articles = pubmed_fetch_articles(pmids, tool=args.tool, email=args.email, api_key=args.api_key)
    by_pmid = {a.pmid: a for a in articles}
    for pmid in pmids:
        a = by_pmid.get(pmid)
        if not a:
            raise PubMedError(f"PMID not returned by PubMed: {pmid}")
        print(_format_ref(a, style=args.style))


def cmd_apply(args: argparse.Namespace) -> None:
    manuscript_path = args.manuscript
    refs_path = args.references
    out_path = args.out or manuscript_path
    inplace = args.inplace

    if inplace and args.out:
        raise ValueError("Use either --inplace or --out, not both.")

    key_to_spec = _load_refs_yaml(refs_path)
    md = manuscript_path.read_text(encoding="utf-8")
    cite_keys = extract_cite_keys(md)
    if not cite_keys:
        raise ValueError(f"No citation tags found in {manuscript_path}. Expected {{{{cite:Key}}}} tags.")

    missing = sorted(set(cite_keys) - set(key_to_spec.keys()))
    if missing:
        raise ValueError(f"Missing citation specs for keys: {missing}")

    key_to_pmid = _resolve_specs_to_pmids(key_to_spec, email=args.email, tool=args.tool, api_key=args.api_key)

    # Number in first-appearance order, de-duplicated.
    ordered_unique_keys: list[str] = []
    seen: set[str] = set()
    for k in cite_keys:
        if k in seen:
            continue
        seen.add(k)
        ordered_unique_keys.append(k)

    key_to_number = {k: i + 1 for i, k in enumerate(ordered_unique_keys)}

    articles = pubmed_fetch_articles(
        [key_to_pmid[k] for k in ordered_unique_keys],
        tool=args.tool,
        email=args.email,
        api_key=args.api_key,
    )
    by_pmid = {a.pmid: a for a in articles}

    # Build references list in citation order.
    refs_lines: list[str] = []
    lock_entries: list[dict[str, Any]] = []
    for k in ordered_unique_keys:
        pmid = key_to_pmid[k]
        a = by_pmid.get(pmid)
        if not a:
            raise PubMedError(f"PMID not returned by PubMed: {pmid} (key={k})")
        refs_lines.append(f"{key_to_number[k]}. {_format_ref(a, style=args.style)}")
        lock_entries.append({"key": k, "number": key_to_number[k], "pmid": pmid, "article": asdict(a)})

    updated = replace_cite_tags(md, key_to_number)

    # Replace or append a References section.
    marker = "\n## References"
    if marker in updated:
        head, _ = updated.split(marker, 1)
        updated = head.rstrip() + "\n\n## References\n\n" + "\n".join(refs_lines) + "\n"
    else:
        updated = updated.rstrip() + "\n\n## References\n\n" + "\n".join(refs_lines) + "\n"

    if inplace:
        manuscript_path.write_text(updated, encoding="utf-8")
        out_path = manuscript_path
    else:
        out_path.write_text(updated, encoding="utf-8")

    if args.lockfile:
        lock = {
            "manuscript": str(out_path),
            "references_yaml": str(refs_path),
            "entries": lock_entries,
        }
        args.lockfile.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {out_path}")
    if args.lockfile:
        print(f"Wrote: {args.lockfile}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="PubMed citation helper: search PubMed, format references, and apply {{cite:Key}} tags."
    )
    p.add_argument("--email", type=str, default=None, help="Contact email for NCBI E-utilities (recommended).")
    p.add_argument("--tool", type=str, default="adni-analysis", help="Tool name for NCBI E-utilities.")
    p.add_argument("--api-key", type=str, default=None, help="NCBI API key (optional).")
    p.add_argument(
        "--style",
        type=str,
        choices=["ama", "nature"],
        default="ama",
        help="Reference style (ama: JAMA; nature: Nature Portfolio/npj).",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="Search PubMed and print PMID + formatted reference.")
    p_search.add_argument("query", type=str, help="PubMed query string (eg, DOI[DOI] or keywords).")
    p_search.add_argument("--retmax", type=int, default=10, help="Max results.")
    p_search.set_defaults(func=cmd_search)

    p_fmt = sub.add_parser("format", help="Format one or more PMIDs as references.")
    p_fmt.add_argument("pmids", type=str, help="Comma-separated PMIDs.")
    p_fmt.set_defaults(func=cmd_format)

    p_apply = sub.add_parser("apply", help="Apply citations to a markdown manuscript using {{cite:Key}} tags.")
    p_apply.add_argument("--manuscript", type=Path, required=True, help="Input markdown file.")
    p_apply.add_argument("--references", type=Path, required=True, help="YAML file mapping citation keys to PMID/DOI.")
    p_apply.add_argument("--out", type=Path, default=None, help="Output markdown path (default: same as input).")
    p_apply.add_argument("--inplace", action="store_true", help="Edit the manuscript in place.")
    p_apply.add_argument(
        "--lockfile",
        type=Path,
        default=None,
        help="Write a JSON lockfile with fetched PubMed metadata and numbering.",
    )
    p_apply.set_defaults(func=cmd_apply)

    args = p.parse_args()
    try:
        args.func(args)
    except PubMedError as e:
        print(f"PubMed error: {e}", file=sys.stderr)
        raise SystemExit(2) from e


if __name__ == "__main__":
    main()
