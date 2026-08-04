# Document toolchain

The release PDF and source-rebuild check were verified with:

- MiKTeX 25.12;
- pdfTeX 3.141592653-2.6-1.40.28;
- BibTeX 0.99e;
- IEEEtran class 1.8b (2015/08/26);
- Python 3.12.5;
- pypdf 6.14.2.

`python tools/build_manuscript.py .` runs pdfLaTeX, BibTeX, and two final pdfLaTeX passes in a fresh temporary directory; rejects undefined citations/references, overfull boxes, and LaTeX errors; and compares normalized text, page count, page geometry, title, and author metadata with the committed PDF. PDF bytes are not required to match because TeX metadata can vary across environments.
