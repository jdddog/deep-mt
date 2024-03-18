#!/usr/bin/env bash

# Define the list of items
data_path="./data/configs/clinical"
items=(
  "logistic-regression-mrs02-36-combined-baseline.yaml"
  "logistic-regression-mrs02-36-no-basilars-combined-baseline.yaml"
  "logistic-regression-sex-no-imaging.yaml"
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