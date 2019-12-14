#!/bin/bash
echo ">>> Kicking off AWS training run for job $1"
git checkout -b train/$1
git push -u origin train/$1
git checkout master
echo ">>> Kick off for job $1 complete."
