#!/bin/bash

bibtex interim
pdflatex interim.tex
bibtex interim
pdflatex interim.tex
