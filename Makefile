# Makefile
SHELL = /bin/bash

.PHONY: help
help:
    @echo "Commands:"
    @echo "style   : executes style formatting."

# Styling
.PHONY: style
style:
	black . && \
	isort . && \
	flake8 
