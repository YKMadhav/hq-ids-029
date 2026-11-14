"""The manuscript may contain no typed result numbers: results enter only via generated macros/tables."""
import json
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(REPO, "paper", "paper.tex")


def _prose_lines():
    for line in open(TEX, encoding="utf-8"):
        body = line.split("%")[0]
        if "\\bibitem" in body or "\\documentclass" in body or "allow-literal" in line:
            continue
        yield body


# Hyperparameters / environment constants are configuration, not results.
ALLOWED = [r"learning rate 0\.\d+", r"\$p=0\.5\$", r"Python 3\.11", r"margin=0\.8in"]


def test_no_typed_decimal_results_in_manuscript():
    def scrub(line):
        for pattern in ALLOWED:
            line = re.sub(pattern, "", line)
        return line

    offenders = [l.strip() for l in _prose_lines() if re.search(r"(?<![\\\w])\d+\.\d+", scrub(l))]
    assert not offenders, offenders[:3]


def test_every_macro_used_is_generated():
    text = open(TEX, encoding="utf-8").read()
    gen = open(os.path.join(REPO, "paper", "generated_numbers.tex"), encoding="utf-8").read()
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", gen))
    used = {m for m in re.findall(r"\\([A-Za-z]+)\{\}", text) if m[0].islower()}
    result_like = {m for m in used if m in defined or re.match(r"(acc|auc|fpr|params|off|pFour|sample|shuffle|thread|noise|mae|mit|boot|perm|topFeat|p[A-Z]|d[A-Z])", m)}
    assert result_like <= defined, sorted(result_like - defined)


def test_generated_numbers_match_json():
    gen = open(os.path.join(REPO, "paper", "generated_numbers.tex"), encoding="utf-8").read()
    data = json.load(open(os.path.join(REPO, "results", "PAPER_NUMBERS.json")))
    for key, value in data.items():
        assert f"\\newcommand{{\\{key}}}{{{value}}}" in gen
