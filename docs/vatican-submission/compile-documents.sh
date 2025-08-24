#!/bin/bash

echo "Compiling Vatican Submission Documents..."

echo
echo "Compiling Cover Letter..."
cd docs/vatican-submission
pdflatex cover-letter.tex
bibtex cover-letter
pdflatex cover-letter.tex
pdflatex cover-letter.tex

echo
echo "Compiling Executive Summary..."
pdflatex executive-summary.tex
bibtex executive-summary
pdflatex executive-summary.tex
pdflatex executive-summary.tex

echo
echo "Compiling Main Divine Necessity Proof..."
cd ../fundamental
pdflatex divine-necessity-mathematical-proof.tex
bibtex divine-necessity-mathematical-proof
pdflatex divine-necessity-mathematical-proof.tex
pdflatex divine-necessity-mathematical-proof.tex

echo
echo "Compilation complete! Check for PDF files in the respective directories."
echo "Note: Some warning messages are normal during LaTeX compilation."