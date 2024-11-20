# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
BUILDDIR      = build
SPHINXGEN     ?= sphinx-apidoc
GENDIR        = generated
CONTENT       = content
STATICDIR     = static
TEMPLATESDIR  = templates

html: flap.rst .deps
	cp -R "$(STATICDIR)" "$(GENDIR)"
	@$(SPHINXBUILD) -M html "$(GENDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.deps:
	mkdir -p $(GENDIR)
	mkdir -p $(BUILDDIR)
	cp conf.py $(GENDIR)
	cp -r $(CONTENT)/* $(GENDIR)

flap.rst: .deps
	$(SPHINXGEN) -eTf -t "$(TEMPLATESDIR)" -o "$(GENDIR)" ../flap ../flap/tools.py

clean:
	rm -rvf "$(BUILDDIR)" "$(GENDIR)"
