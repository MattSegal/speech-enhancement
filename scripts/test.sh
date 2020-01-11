#!/bin/bash
if [[ -d "env" ]]; then
    . ./env/bin/activate
fi
if [[ -z "$@" ]]; then
    python -W ignore -m pytest tests
else
    python -W ignore -m pytest $@
fi
