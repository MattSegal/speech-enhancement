#! /bin/bash
if [[ -d "env" ]]; then
    . ./env/bin/activate
fi
python -W ignore -m src dev