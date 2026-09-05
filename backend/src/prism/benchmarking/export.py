"""Deterministic export of benchmark reports to JSON, Markdown, and CSV."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from prism.benchmarking.contracts import (
    BenchmarkMatrix,
    BenchmarkTable,
    PRISMResearchReport,
)


def export_report_to_json(
    report: PRISMResearchReport,
    output_path: str | Path | None = None,
    indent: int = 2,
) -> str:
    """Export complete research report to deterministic formatted JSON."""
    data = report.to_dict()
    json_str = json.dumps(data, indent=indent, sort_keys=True)
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_str, encoding="utf-8")
    return json_str


def export_table_to_csv(table: BenchmarkTable) -> str:
    """Export a BenchmarkTable to CSV string format."""
    output = io.StringIO()
    writer = csv.writer(output)

    if not table.rows:
        return ""

    col_keys = [k for k in table.rows[0] if k != table.row_factor]
    headers = [table.row_factor, *col_keys]
    writer.writerow(headers)

    for row_dict in table.rows:
        row_val = str(row_dict.get(table.row_factor, ""))
        vals = [row_val]
        for col in col_keys:
            cell_info = row_dict.get(col)
            if isinstance(cell_info, dict):
                disp = str(cell_info.get("display", cell_info.get("value", "")))
            else:
                disp = str(cell_info if cell_info is not None else "")
            vals.append(disp)
        writer.writerow(vals)

    return output.getvalue()


def export_matrix_to_csv(matrix: BenchmarkMatrix) -> str:
    """Export a BenchmarkMatrix to CSV string format."""
    output = io.StringIO()
    writer = csv.writer(output)

    headers = [matrix.row_factor, *matrix.column_values]
    writer.writerow(headers)

    for r_val in matrix.row_values:
        row = [r_val]
        for c_val in matrix.column_values:
            cell = matrix.cells.get(r_val, {}).get(c_val)
            val_str = ""
            if cell is not None and hasattr(cell, "value") and cell.value is not None:
                val_str = str(cell.value)
            elif cell is not None and hasattr(cell, "mean") and cell.mean is not None:
                val_str = str(cell.mean)
            row.append(val_str)
        writer.writerow(row)

    return output.getvalue()


def export_report_to_markdown(
    report: PRISMResearchReport,
    output_path: str | Path | None = None,
) -> str:
    """Export research report to publication-grade GitHub-flavored Markdown."""
    lines: list[str] = []

    lines.append(f"# {report.title}")
    lines.append("")
    lines.append(f"**Report ID:** `{report.report_id}`  ")
    lines.append(f"**Campaign ID:** `{report.campaign_id}`  ")
    fp = report.reproducibility_manifest.campaign_fingerprint
    lines.append(f"**Campaign Fingerprint:** `{fp}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"> {report.executive_summary}")
    lines.append("")

    # Methodology Summary
    lines.append("## Experimental Methodology")
    lines.append("")
    lines.append(report.methodology_summary)
    lines.append("")

    if report.warnings:
        lines.append("> [!WARNING]")
        for w in report.warnings:
            lines.append(f"> - {w}")
        lines.append("")

    # Benchmark Tables
    if report.tables:
        lines.append("## Benchmark Result Tables")
        lines.append("")
        for tbl in report.tables:
            lines.append(f"### {tbl.title}")
            lines.append("")
            if tbl.rows:
                col_keys = [k for k in tbl.rows[0] if k != tbl.row_factor]
                headers = [
                    tbl.row_factor.title(),
                    *[c.title() for c in col_keys],
                ]
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row_dict in tbl.rows:
                    row_val = str(row_dict.get(tbl.row_factor, "")).title()
                    vals = [row_val]
                    for col in col_keys:
                        cell_info = row_dict.get(col)
                        if isinstance(cell_info, dict):
                            disp = str(
                                cell_info.get("display", cell_info.get("value", "—"))
                            )
                        else:
                            disp = str(cell_info if cell_info is not None else "—")
                        vals.append(disp)
                    lines.append("| " + " | ".join(vals) + " |")
                lines.append("")

    # Research Findings
    if report.findings:
        lines.append("## Scientific Findings Grounded in Observed Evidence")
        lines.append("")
        for f in report.findings:
            lines.append(f"### Finding `{f.finding_id}`")
            lines.append(f"**Evidence Strength:** `{f.evidence_strength.value}`")
            lines.append("")
            lines.append(f"> {f.statement}")
            lines.append("")
            if f.caveats:
                lines.append("**Caveats & Limitations:**")
                for c in f.caveats:
                    lines.append(f"- {c}")
                lines.append("")

    # Evidence Gaps
    if report.evidence_gaps:
        lines.append("## Evidence Gaps & Missing Experiments")
        lines.append("")
        for gap in report.evidence_gaps:
            lines.append(f"- **[{gap.gap_id}]** {gap.rationale}")
        lines.append("")

    # Reproducibility Appendix
    lines.append("## Reproducibility Appendix")
    lines.append("")
    total_reg = report.reproducibility_manifest.environment_provenance.get(
        "total_registered_results", 0
    )
    lines.append(f"- **Total Registered Observations:** {total_reg}")
    lines.append(
        f"- **Registered Random Seeds:** {report.reproducibility_manifest.seeds}"
    )
    fp_cnt = len(report.reproducibility_manifest.experiment_fingerprints)
    lines.append(f"- **Unique Fingerprints:** {fp_cnt}")
    lines.append("")

    md_content = "\n".join(lines)
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md_content, encoding="utf-8")

    return md_content
