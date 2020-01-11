#!/bin/bash
if [[ -d "env" ]]; then
    . ./env/bin/activate
fi
python3.6 scripts/infra $@