**The six-month window has passed.** `qbmkit` went public on **2026-08-07**, so as of
today it clears the JOSS pre-review maturity gate that blocked submission back then:

> "Repository made public 6+ months ago with clear evidence of ongoing development"

That was the *only* thing standing in the way — everything else on the checklist was
already satisfied.

## Before you submit, confirm the gate is genuinely cleared

The date is necessary but not sufficient. JOSS also rejects submissions where
"all or most commits [are] concentrated in the last few weeks before submission".

- [ ] Commits are **distributed across the six months**, not bunched — check the
      contribution graph
- [ ] Some of the open [issues](../../issues) have been closed, ideally a few by other
      people
- [ ] There is evidence the software is **used for research** — a paper of yours that
      cites or uses it, or an external user
- [ ] `README`, `DESIGN.md` and `paper/paper.md` still describe what the code actually
      does (week 24 of the [codebase tour](../../blob/main/docs/codebase-tour.md))
- [ ] The full suite passes and CI is green on every supported Python

## Then

- [ ] Cut a fresh release so the Zenodo archive matches what reviewers will read
- [ ] Update `version:` and `date-released:` in `CITATION.cff` and `.zenodo.json`
- [ ] Replace the placeholder preprint citation in `paper/paper.bib` (issue #11) if it
      is still outstanding
- [ ] Write the **generative-AI disclosure** — JOSS requires it: the tools and versions
      used, the nature and scope of the assistance, and an assertion that you reviewed,
      edited and validated all AI-assisted output. Non-disclosure is treated as an
      ethical breach
- [ ] Preview the paper: **Actions → Draft JOSS paper → Run workflow**, then download the
      PDF artifact
- [ ] Submit at <https://joss.theoj.org/papers/new> with the repository URL, the
      **concept** DOI `10.5281/zenodo.21844741`, and the released version

## Reference

- Submission requirements: <https://joss.readthedocs.io/en/latest/submitting.html>
- Review criteria: <https://joss.readthedocs.io/en/latest/review_criteria.html>

Close this issue once submitted.
