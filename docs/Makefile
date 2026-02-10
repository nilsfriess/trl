# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
DOXYGEN       = doxygen

.PHONY: help clean html doxygen all

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  doxygen    to generate Doxygen XML"
	@echo "  all        to generate both Doxygen XML and HTML"
	@echo "  clean      to remove generated files"

clean:
	rm -rf $(BUILDDIR)/* doxygen/

doxygen:
	cd .. && $(DOXYGEN) Doxyfile

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

all: doxygen html
