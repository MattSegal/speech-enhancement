#! /bin/bash
if [[ -d "env" ]]; then
    . ./env/bin/activate
fi
python3.6 -W ignore -m src dev