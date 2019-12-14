#!/bin/bash
if [[ -d "env" ]]; then
    . ./env/bin/activate
fi
python3 -m src.datasets.speech.speech_evaluation.evaluate
