#!/bin/bash
echo -e ">>> Kicking off AWS training run for job $1\n"
git checkout -b train/$1
git push -u origin train/$1
git checkout master
echo -e "\n>>> Kick off for job $1 complete.\n"
