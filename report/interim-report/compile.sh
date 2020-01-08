#!/bin/bash

latex interim
bibtex interim
latex interim
bibtex interim
latex interim
pdflatex interim.tex
