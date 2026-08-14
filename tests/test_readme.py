"""Tests for README.md (F5.S2.T2, Spec F5 acceptance criterion 2)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
TEXT = README.read_text(encoding="utf-8")

# What a reader actually sees: HTML comments are stripped, whitespace collapsed
# so an assertion doesn't depend on where a line happens to wrap.
VISIBLE = " ".join(re.sub(r"<!--.*?-->", "", TEXT, flags=re.DOTALL).split())

SHIPPED_DOCS = [
    ROOT / "README.md",
    ROOT / "REPRODUCE.md",
    ROOT / "REFERENCE.md",
    ROOT / ".claude" / "skills" / "kleidi-advisor" / "SKILL.md",
]

ATTRIBUTION_LINE = (
    "Speedup comes from Arm's KleidiAI kernels; this tool detects the miss and measures the delta."
)

SECTION_HEADINGS = [
    "## 1. The Finding",
    "## 2. Why Nobody Notices",
    "## 3. What This Is / Is Not",
    "## 4. Quickstart",
    "## 5. Results",
    "## 6. How It Works",
    "## 7. Verify It Yourself",
    "## 8. What We Got Wrong",
    "## 9. Limitations",
    "## 10. Future Work",
    "## 11. License",
]


def test_ten_sections_present_in_order():
    positions = [TEXT.find(heading) for heading in SECTION_HEADINGS]
    for heading, pos in zip(SECTION_HEADINGS, positions):
        assert pos != -1, f"missing heading {heading!r}"
    assert positions == sorted(positions), "sections are not in the required order"


def test_no_placeholders_remain():
    # Every results surface is measured now, so the placeholder budget is zero.
    # Built by concatenation so this assertion never matches itself.
    token = "TODO" + "(box)"
    assert token not in TEXT, f"{TEXT.count(token)} unfilled placeholder(s) left in README.md"


def test_headline_pairs_throughput_with_its_quality_cost():
    # The no-bare-throughput rule, checked on the shipped prose rather than
    # only on `report`'s output.
    results = TEXT[TEXT.find("## 5. Results"):TEXT.find("## 6. How It Works")]
    assert "1.61×" in results
    assert "+0.049" in results
    assert "100 chunks" in results


def test_quality_cost_is_not_overclaimed_as_equivalence():
    # +0.049 is smaller than the ±0.14 error bars, which limits what this run
    # can resolve — it is not evidence that the two models are the same.
    for banned in ("identical quality", "no quality cost", "quality is unchanged", "lossless"):
        assert banned.lower() not in TEXT.lower(), f"README overclaims: {banned!r}"
    assert "inside the error bars" in TEXT


def test_limitations_names_every_axis_the_run_did_not_cover():
    limitations = TEXT[TEXT.find("## 9. Limitations"):TEXT.find("## 10. Future Work")]
    for required in ("8-vCPU", "Neoverse-N2", "Qwen2.5-7B-Instruct", "b10431",
                     "--chunks 100", "hand-assembled", "not a"):
        assert required in limitations, f"Limitations never states {required!r}"


def test_attribution_line_present_verbatim():
    assert ATTRIBUTION_LINE in TEXT


def test_quickstart_has_at_most_five_commands():
    start = TEXT.find("## 4. Quickstart")
    end = TEXT.find("## 5. Results")
    quickstart = TEXT[start:end]
    fenced = re.search(r"```bash\n(.*?)```", quickstart, re.DOTALL)
    assert fenced, "expected a fenced bash block in the Quickstart section"
    commands = [line for line in fenced.group(1).splitlines() if line.strip()]
    assert 1 <= len(commands) <= 5, f"expected 1-5 Quickstart commands, found {len(commands)}"


# --- Claims that must survive an Arm engineer reading them -------------------

# llama.cpp *does* warn: on b10431 a Q4_K_M load logs
#   kleidiai: no kernel for tensor type q4_K, not accelerated by KleidiAI
# Any claim that the miss is unlogged is factually wrong and, worse, is wrong
# in a way that a reader can disprove in one command — which discredits the
# measured claims sitting next to it. The honest framing (the warning is real
# but arrives post-download, unranked and uncosted) is also the stronger one.
UNLOGGED_MISS_CLAIMS = [
    "no error, no warning",
    "no warning",
    "without warning",
    "without any warning",
    "silent miss",
    "the miss is silent",
    "silently falls back",
    "silently missing",
    "silently misses",
    "silently runs",
    "no log line",
    "nothing in the log",
    "nothing in the logs",
]


def test_no_doc_claims_the_miss_goes_unlogged():
    for path in SHIPPED_DOCS:
        haystack = " ".join(path.read_text(encoding="utf-8").split()).lower()
        for claim in UNLOGGED_MISS_CLAIMS:
            assert claim not in haystack, (
                f"{path.name} claims the miss is unlogged ({claim!r}); llama.cpp does warn — "
                "see README §2 and the §7 dispatch evidence"
            )


def test_readme_quotes_the_actual_warning_and_says_why_it_is_not_enough():
    why = TEXT[TEXT.find("## 2. Why Nobody Notices"):TEXT.find("## 3. What This Is")]
    assert "no kernel for tensor type q4_K, not accelerated by KleidiAI" in why
    # The three reasons it still isn't actionable.
    assert "load time" in why
    assert "before the download" in why.lower() or "before you download" in why.lower()
    for cost_word in ("quantifies nothing", "1.61×"):
        assert cost_word in why, f"§2 never says the warning carries no cost figure ({cost_word})"


# --- Provenance: no claim about results/ that results/ doesn't back ----------


def _bench_results_exist() -> bool:
    """True once `kleidi-advisor bench` has written real result files."""
    from kleidi_advisor.report import load_results

    return bool(load_results(ROOT / "results"))


def test_report_provenance_claim_is_hidden_until_results_actually_exist():
    """Arbitrates the claim instead of trusting anyone to remember it.

    README §5 may say the section is rendered by `report` from `results/*.json`
    only when that is true. Until then the sentence lives in an HTML comment,
    where it renders as nothing. Uncommenting it without producing the results
    files fails here.
    """
    claim_visible = "renders from `results/*.json`" in VISIBLE
    if claim_visible:
        assert _bench_results_exist(), (
            "README §5 claims to be rendered by `report` from results/*.json, but no schema-1 "
            "bench result exists there. Produce them with `kleidi-advisor bench`, or re-comment "
            "the sentence (see the PENDING-PROVENANCE marker in README.md §5)."
        )


def test_results_section_states_how_the_numbers_were_actually_produced():
    assert "by hand" in VISIBLE
    assert "llama-bench" in VISIBLE and "llama-perplexity" in VISIBLE


def test_no_plot_is_referenced_until_one_exists():
    if "results/plot.png" in VISIBLE:
        assert (ROOT / "results" / "plot.png").exists(), (
            "README references results/plot.png but no plot has been generated"
        )


# --- Attribution of the measured rows ----------------------------------------


def test_published_and_our_own_rows_are_attributed_to_the_right_authors():
    results = TEXT[TEXT.find("## 5. Results"):TEXT.find("## 6. How It Works")]
    normalized = " ".join(results.split())

    assert "published-q4_0" in results
    assert "from the same repository as the Q4_K_M baseline" in normalized
    assert "The `imatrix-fix` row is *our* artifact" in normalized
    # No row may be tagged plain "fixed" while it is really Qwen's own file.
    for line in results.splitlines():
        if line.startswith("| qwen-q4_0-ref"):
            assert "| fixed" not in line, line
            assert "published-q4_0" in line, line


def test_fix_row_carries_its_own_measurements_not_the_published_build_s():
    """Was: assert the row stays marked pending. Now that it is measured, the
    same risk points the other way — a fix row silently carrying the published
    build's numbers would read as success and be wrong."""
    results = TEXT[TEXT.find("## 5. Results"):TEXT.find("## 6. How It Works")]
    table_rows = [line for line in results.splitlines() if line.startswith("| qwen")]
    fix_rows = [line for line in table_rows if "imatrix-fix" in line and "±" in line]
    assert len(fix_rows) == 1, "expected exactly one imatrix-fix data row"
    row = fix_rows[0]
    assert "_pending_" not in row, "the fix row is measured; it must not still say pending"
    assert "66.65" in row and "17.13" in row and "8.1525" in row
    # The published build's figures must not appear in our row.
    for published in ("71.60", "17.61", "8.2215"):
        assert published not in row, f"fix row carries the published build's {published}"


# --- Internal consistency ----------------------------------------------------


def test_no_command_calibrates_against_an_unshipped_corpus_file():
    """Bans the invocation, not the word.

    Every `--calib` must name the file the documented download actually
    produces. README §5 now *discusses* wiki.train.raw as the remedy for the
    calibration overlap, which is prose, not a command a reader would paste.
    """
    for path in SHIPPED_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "--calib wiki.train.raw" not in text, f"{path.name} calibrates on an unshipped file"
        assert "-f wiki.train.raw" not in text, f"{path.name} calibrates on an unshipped file"


def test_measured_ratio_never_appears_without_its_quality_cost_on_the_same_line():
    """Per *physical line*, not per document.

    A reader or an agent quoting one line must not come away with a bare
    speedup. Word-wrapping a sentence between the ratio and its ppl delta is
    the failure mode this catches — it has happened twice in this repo.
    """
    for i, line in enumerate(TEXT.splitlines(), start=1):
        if "1.61" not in line or "×" not in line:
            continue
        assert "ppl" in line or "perplexity" in line or "0.049" in line, (
            f"README.md:{i} quotes the measured ratio with no quality cost on the same line: "
            f"{line.strip()!r}"
        )


# --- The two caveats on our own fix row --------------------------------------
#
# Both must sit next to the number they qualify. A caveat parked in §9 is one a
# reader reaches after they have already quoted the row.

RESULTS_START = "## 5. Results"
RESULTS_END = "## 6. How It Works"


def _results_section() -> str:
    return TEXT[TEXT.find(RESULTS_START):TEXT.find(RESULTS_END)]


# Claiming imatrix improved quality would be reading a contaminated number as a
# result: the matrix was calibrated on wiki.test.raw and perplexity evaluated on
# wiki.test.raw, so a -0.020 "gain" is exactly what calibrating on the eval set
# would produce whether or not imatrix helps.
QUALITY_GAIN_CLAIMS = [
    "imatrix improves quality",
    "improves quality",
    "quality improvement",
    "better quality",
    "improves perplexity",
    "lower perplexity than",
    "quality gain",
]


def test_contamination_caveat_sits_with_the_fix_row_not_in_limitations():
    results = _results_section()
    normalized = " ".join(results.split())

    assert "qwen-imatrix-q4_0" in results, "the fix row is missing"
    # Names the overlap explicitly, both sides of it.
    assert "calibrated on `wiki.test.raw`" in normalized
    assert "evaluated on `wiki.test.raw`" in normalized
    assert "the same file" in normalized
    # Names the remedy and why it wasn't taken.
    assert "wiki.train.raw" not in normalized or "would remove the overlap" in normalized
    # And says plainly what the number may not be used for.
    assert "not evidence that imatrix improves quality" in normalized


def test_no_doc_reads_the_contaminated_delta_as_a_quality_win():
    for path in SHIPPED_DOCS:
        haystack = " ".join(path.read_text(encoding="utf-8").split()).lower()
        for claim in QUALITY_GAIN_CLAIMS:
            if claim not in haystack:
                continue
            # The only permitted use is denying it.
            idx = haystack.find(claim)
            window = haystack[max(0, idx - 60):idx]
            assert "not evidence that" in window or "must not" in window, (
                f"{path.name} presents the contaminated ppl delta as a quality win: "
                f"...{haystack[max(0, idx - 60):idx + 40]}..."
            )


def test_unexplained_speed_gap_is_stated_as_open_and_not_speculated_past():
    results = _results_section()
    normalized = " ".join(results.split())

    # The gap itself, with both numbers so a reader can check the arithmetic.
    assert "6.9% slower at pp512" in normalized
    assert "66.65" in normalized and "71.60" in normalized
    # Stated as unexplained, and explicitly not investigated.
    assert "unexplained" in normalized.lower()
    assert "did not investigate" in normalized.lower()
    # Causes are named as candidates, never asserted.
    assert "not as conclusions" in normalized or "as candidates" in normalized
    for asserted_cause in ("this is because", "the cause is", "caused by", "due to the"):
        assert asserted_cause not in normalized.lower(), (
            f"§5 asserts a cause for the unexplained gap ({asserted_cause!r}); it was not investigated"
        )


def test_fix_row_is_never_presented_as_beating_the_published_build():
    normalized = " ".join(TEXT.split()).lower()
    for overclaim in ("faster than qwen's", "beats the published", "outperforms the published"):
        assert overclaim not in normalized, f"README overclaims the fix row: {overclaim!r}"


def test_headline_ratio_still_comes_from_the_two_published_builds():
    results = _results_section()
    headline_row = [
        line for line in results.splitlines()
        if line.startswith("| **`published-q4_0`**")
    ]
    assert len(headline_row) == 1, "the headline ratio row must name the published build"
    assert "1.61×" in headline_row[0]
    # Our own fix row must be labelled as ours and must not carry the headline.
    fix_row = [line for line in results.splitlines() if line.startswith("| `imatrix-fix`")]
    assert len(fix_row) == 1
    assert "1.50×" in fix_row[0] and "1.61×" not in fix_row[0]


# --- §5 must stay what `report` actually renders ------------------------------


def test_results_section_matches_what_report_renders_from_results_dir():
    """The claim in §5 is that the block is `report`'s output. This checks it
    against the real thing rather than trusting the transcription — a stale
    paste is exactly the kind of drift that turns a verifiable claim into a
    false one."""
    from kleidi_advisor.report import load_results, render_markdown

    entries = load_results(ROOT / "results")
    assert entries, "results/ has no schema-1 bench files; §5 cannot claim to be rendered"

    rendered = render_markdown(
        entries,
        instance="Azure Standard_E8ps_v6 (Cobalt 100, Neoverse N2), 8 threads",
        headline_tag="published-q4_0",
    )
    lines = rendered.splitlines()

    # Headline verbatim.
    assert lines[0] in TEXT, f"README headline is stale; report now renders: {lines[0]}"
    # Every data row's cells, allowing README to pad the table for readability.
    for line in lines:
        if not line.startswith("| qwen"):
            continue
        for cell in (c.strip() for c in line.strip("|").split("|")):
            assert cell in TEXT, f"§5 is missing rendered cell {cell!r} from row {line!r}"


def test_every_result_file_records_its_instance_and_commit():
    import json

    files = sorted((ROOT / "results").glob("*.json"))
    bench_files = [
        f for f in files
        if isinstance(json.loads(f.read_text(encoding="utf-8")).get("metrics"), dict)
    ]
    assert len(bench_files) == 3, f"expected three bench result files, found {len(bench_files)}"
    for path in bench_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for field in ("instance", "llama_cpp_commit"):
            value = data.get(field)
            assert value, f"{path.name}: {field} is empty"
            assert "TODO" not in value, f"{path.name}: {field} still holds a placeholder"
        assert "Neoverse N2" in data["instance"], path.name
        assert "1692f9e50" in data["llama_cpp_commit"], path.name


# --- §1 must not drift from §5 ------------------------------------------------
#
# §1 and §5 quote the same three measurements. §5 is rendered from results/ and
# so cannot drift; §1 is hand-written prose, and it did drift — it carried
# hand-run figures for two full turns after §5 became rendered output. This
# binds §1's table to the same source of truth.

# §1 labels its rows by quantization format; results/ tags them by role.
SECTION_1_ROW_TO_TAG = {"Q4_K_M": "baseline", "Q4_0": "published-q4_0"}


def _rendered_cells_by_tag() -> dict:
    """{tag: {"pp512": "44.47 ± 0.04", "tg128": ..., "ppl": "8.1728"}}"""
    from kleidi_advisor.report import load_results, render_markdown

    rendered = render_markdown(
        load_results(ROOT / "results"),
        instance="Azure Standard_E8ps_v6 (Cobalt 100, Neoverse N2), 8 threads",
        headline_tag="published-q4_0",
    )
    by_tag = {}
    for line in rendered.splitlines():
        if not line.startswith("| qwen"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        _model, tag, _threads, pp512, tg128, ppl = cells
        by_tag[tag] = {"pp512": pp512, "tg128": tg128, "ppl": ppl}
    return by_tag


def _section_1_rows() -> dict:
    finding = TEXT[TEXT.find("## 1. The Finding"):TEXT.find("## 2. Why Nobody Notices")]
    rows = {}
    for line in finding.splitlines():
        if not line.startswith("| Q4"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        fmt, pp512, tg128, ppl = cells[0], cells[1], cells[2], cells[3]
        rows[fmt] = {"pp512": pp512, "tg128": tg128, "ppl": ppl}
    return rows


def test_section_1_table_matches_the_rendered_values_in_section_5():
    rendered = _rendered_cells_by_tag()
    section_1 = _section_1_rows()
    assert set(section_1) == set(SECTION_1_ROW_TO_TAG), f"unexpected §1 rows: {sorted(section_1)}"

    for fmt, tag in SECTION_1_ROW_TO_TAG.items():
        for metric in ("pp512", "tg128"):
            assert section_1[fmt][metric] == rendered[tag][metric], (
                f"§1 row {fmt} {metric} is {section_1[fmt][metric]!r} but results/ renders "
                f"{rendered[tag][metric]!r} — §1 has drifted from §5"
            )
        # §1 adds llama-perplexity's own uncertainty, which the results schema
        # does not store; the value itself must still agree.
        assert section_1[fmt]["ppl"].split("±")[0].strip() == rendered[tag]["ppl"], (
            f"§1 row {fmt} ppl disagrees with results/"
        )


def test_section_1_and_section_5_agree_on_the_derived_ratios():
    rendered = _rendered_cells_by_tag()
    base = float(rendered["baseline"]["pp512"].split("±")[0])
    pub = float(rendered["published-q4_0"]["pp512"].split("±")[0])
    base_tg = float(rendered["baseline"]["tg128"].split("±")[0])
    pub_tg = float(rendered["published-q4_0"]["tg128"].split("±")[0])

    assert f"{pub / base:.2f}×" == "1.61×"
    assert f"{pub_tg / base_tg:.2f}×" == "1.12×"
    # Both ratios must appear in the README exactly as the medians imply.
    for ratio in (f"{pub / base:.2f}×", f"{pub_tg / base_tg:.2f}×"):
        assert ratio in TEXT, f"README never states the derived ratio {ratio}"
    # And the superseded rounding must be gone from the prose.
    finding = TEXT[TEXT.find("## 1. The Finding"):TEXT.find("## 2. Why Nobody Notices")]
    assert "1.11×" not in finding, "§1 still carries the pre-render tg128 ratio"


def test_tg128_sentence_does_not_imply_a_per_metric_perplexity():
    # "1.11x at the same +0.049 ppl" read as though perplexity had been measured
    # once per throughput metric. It is one measurement for the model pair.
    normalized = " ".join(TEXT.split())
    assert "at the same +0.049 ppl" not in normalized
    assert "the +0.049 ppl above is the quality cost for the pair" in normalized
