"""Markdown + HTML report rendering — no task text ever leaks into output."""

from __future__ import annotations

import math

from llm_verdict.reporting.data import (
    CategoryStats,
    CompareReport,
    RunReport,
    TimelineReport,
)


def render_card_markdown(report: RunReport) -> str:
    """Render a single-model report card as Markdown."""
    lines: list[str] = []
    lines.append(f"# Model Report Card: {report.model_id}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Model | {report.model_id} |")
    lines.append(f"| Version | {report.model_version or 'N/A'} |")
    lines.append(f"| Run Date | {report.run_date} |")
    lines.append(f"| Suite Hash | {report.suite_hash[:12]} |")
    lines.append(f"| Harness Version | {report.harness_version} |")
    lines.append(f"| Judge Config | {report.judge_config_hash or 'N/A'} |")
    lines.append(f"| Total Cost | ${report.total_cost:.4f} |")
    lines.append("")

    lines.append("## Results by Category")
    lines.append("")
    lines.append(_category_table(report.categories))
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append(_category_table([report.overall]))
    lines.append("")

    if report.notable_failures:
        lines.append("## Notable Failure Patterns")
        lines.append("")
        for pattern in report.notable_failures:
            lines.append(f"- {pattern}")
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("- Trials per task: inferred from data")
    lines.append("- Temperature: 0.0")
    lines.append("- Judge model family: see judge config")
    n_total = sum(c.n_tasks for c in report.categories)
    if n_total > 0:
        for c in report.categories:
            pct = c.n_tasks / n_total * 100
            lines.append(f"- {c.category}: {pct:.0f}% of tasks")
    lines.append("")

    return "\n".join(lines)


def render_card_html(report: RunReport) -> str:
    """Wrap Markdown report in a self-contained HTML page."""
    md = render_card_markdown(report)
    return _wrap_html(f"Report Card: {report.model_id}", md)


def render_compare_markdown(report: CompareReport) -> str:
    """Render a head-to-head comparison as Markdown."""
    lines: list[str] = []
    lines.append(f"# Head-to-Head: {report.run_a.model_id} vs {report.run_b.model_id}")
    lines.append("")
    lines.append(f"| | {report.run_a.model_id} | {report.run_b.model_id} |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Pass Rate | {report.run_a.overall.pass_rate.point:.1%} | "
        f"{report.run_b.overall.pass_rate.point:.1%} |"
    )
    lines.append(
        f"| Cost | ${report.run_a.total_cost:.4f} | ${report.run_b.total_cost:.4f} |"
    )
    lines.append("")

    lines.append("## Per-Category Deltas (B - A)")
    lines.append("")
    lines.append("| Category | Delta | CI | McNemar p |")
    lines.append("|----------|-------|----|-----------|")
    for cat, delta in report.delta_by_category.items():
        mc = report.mcnemar_by_category[cat]
        lines.append(
            f"| {cat} | {delta.point:+.3f} | "
            f"[{delta.lower:+.3f}, {delta.upper:+.3f}] | "
            f"{mc.p_value:.4f} |"
        )
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    d = report.overall_delta
    m = report.overall_mcnemar
    lines.append(f"- Score delta: {d.point:+.3f} [{d.lower:+.3f}, {d.upper:+.3f}]")
    lines.append(f"- McNemar p-value: {m.p_value:.4f} (n_discordant={m.n_discordant})")
    lines.append("")

    return "\n".join(lines)


def render_compare_html(report: CompareReport) -> str:
    """Wrap comparison Markdown in a self-contained HTML page."""
    md = render_compare_markdown(report)
    title = f"Compare: {report.run_a.model_id} vs {report.run_b.model_id}"
    return _wrap_html(title, md)


def _category_table(categories: list[CategoryStats]) -> str:
    lines: list[str] = []
    lines.append(
        "| Category | N | Pass Rate | Mean Score | Cost/Success | "
        "p50 (ms) | p95 (ms) | Refusal % | Consistency |"
    )
    lines.append("|" + "|".join(["---"] * 9) + "|")
    for c in categories:
        cps_val = c.cost_per_success.point
        cps = f"${cps_val:.4f}" if math.isfinite(cps_val) else "N/A"
        lines.append(
            f"| {c.category} | {c.n_tasks} | "
            f"{c.pass_rate.point:.1%} ±{c.pass_rate.width / 2 * 100:.1f}pp | "
            f"{c.mean_score.point:.3f} ±{c.mean_score.width / 2:.3f} | "
            f"{cps} | "
            f"{c.p50_latency_ms:.0f} | {c.p95_latency_ms:.0f} | "
            f"{c.refusal_rate:.1f}% | {c.consistency:.0%} |"
        )
    return "\n".join(lines)


def render_timeline_markdown(report: TimelineReport) -> str:
    """Render a longitudinal timeline as Markdown."""
    lines: list[str] = []
    lines.append(f"# Timeline: {report.model_family}*")
    lines.append("")
    lines.append("| Date | Model | Version | Suite | Pass Rate | Score | Cost |")
    lines.append("|------|-------|---------|-------|-----------|-------|------|")
    for e in report.entries:
        lines.append(
            f"| {e.run_date[:10]} | {e.model_id} | "
            f"{e.model_version or 'N/A'} | {e.suite_hash[:8]} | "
            f"{e.pass_rate:.1%} | {e.mean_score:.3f} | "
            f"${e.total_cost:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _wrap_html(title: str, markdown_body: str) -> str:
    """Minimal self-contained HTML with preformatted Markdown."""
    escaped = (
        markdown_body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: monospace; max-width: 900px; margin: 2em auto; padding: 0 1em; }}
pre {{ white-space: pre-wrap; }}
</style>
</head>
<body>
<pre>{escaped}</pre>
</body>
</html>"""
