#!/usr/bin/env bash

# Define the list of items
data_path="./data/configs/clinical"
items=(
  "logistic-regression-mrs02-36.yaml"
  "logistic-regression-mrs02-36-no-basilars.yaml"
  "random-forest-mrs02-36.yaml"
  "random-forest-mrs02-36-no-basilars.yaml"
  "logistic-regression-sex.yaml"
)

# Trap the SIGINT signal (sent when the user
# presses "ctrl+c" on the keyboard)
trap 'echo "Exiting script"; exit' SIGINT

# Loop through the list of items
for item in ${items[@]}
do
  # Echo the current item
  echo "Running deep-mt feature-selection on $item"
  deep-mt feature-selection $data_path/$item || exit
done