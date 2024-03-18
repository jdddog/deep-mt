#!/usr/bin/env bash

# Define the list of items
data_path="./data/configs/imaging/sex"
items=("sex-ct-152x182x76px.yaml"
       "sex-ct-1.0x1.0x2.0mm-152x182x76px.yaml"
       "sex-ct-ss-1.0x1.0x2.0mm-152x182x76px.yaml"
       "sex-ct-ss-1.0x1.0x2.0mm-152x182x76px-rand-trans.yaml")

# Trap the SIGINT signal (sent when the user
# presses "ctrl+c" on the keyboard)
trap 'echo "Exiting script"; exit' SIGINT

# Loop through the list of items
for item in ${items[@]}
do
  echo "Running deep-mt train-pytorch on $item"
  deep-mt train-pytorch $data_path/$item || exit

  echo "Running deep-mt evaluate-pytorch on $item"
  deep-mt evaluate-pytorch $data_path/$item --subset train --subset valid --subset test || exit
done