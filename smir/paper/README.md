# SMIR paper draft (AAAI / Workshop)

- **Main file:** `smir_aaai_draft.tex`
- **Format:** self-contained two-column `article` approximating AAAI letter layout (works without the official kit).

## Compile

```bash
cd /project/peilab/why/NEW/smir/paper
pdflatex smir_aaai_draft.tex
pdflatex smir_aaai_draft.tex
```

## Switch to official AAAI style

1. Download the AAAI author kit and place `aaai25.sty` / `aaai26.sty` (as required) in this folder.
2. Replace the preamble with the kit’s template preamble (`\usepackage{aaai2x}`, times/helvet/courier, etc.).
3. Keep the body from `\maketitle` onward; shrink floats if the kit’s column constraints differ.

## Workshop vs AAAI length

- **Workshop (≈4 pages):** drop appendix sketches; compress Related Work + Discussion.
- **AAAI Main (≤7 pages + refs):** expand proofs into a technical appendix PDF; add multi-seed tables when `pub_pack.json` is ready.

## Numbers in the draft

Table 1 uses seed-0 gated results + AAAI Hard main + 7B negative from `smir/results/aaai_pack.json`. Update after multiseed pack finishes.
