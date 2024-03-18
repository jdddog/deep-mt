#!/usr/bin/env bash

# Define the list of items
data_path="./data/configs/combined"
items=(
  "logistic-regression-imaging-mrs02-36-combined.yaml"
  "logistic-regression-imaging-mrs02-36-combined-no-basilars.yaml"
)

# Trap the SIGINT signal (sent when the user
# presses "ctrl+c" on the keyboard)
trap 'echo "Exiting script"; exit' SIGINT

# Loop through the list of items
for item in ${items[@]}
do
  # Echo the current item
  echo "Running deep-mt train-sklearn on $item"
  deep-mt train-sklearn $data_path/$item || exit
done