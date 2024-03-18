#!/usr/bin/env bash

# Define the list of items
data_path="./data/configs/imaging/mrs"

items=(
  "mrs-ct-ventricles-0.44x0.44x1.5mm.yaml"
  "mrs-ct-cta-ventricles-0.44x0.44x1.5mm.yaml"
)

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