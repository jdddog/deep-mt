#!/usr/bin/env bash

# Trap the SIGINT signal (sent when the user
# presses "ctrl+c" on the keyboard)
trap 'echo "Exiting script"; exit' SIGINT

# Set the image keys and subsets to be used in the script
image_keys=("ct" "cta")
subsets=("train" "valid" "test" "all")

# Loop through the image keys and subsets, and run the command
for image_key in "${image_keys[@]}"; do
  for subset in "${subsets[@]}"; do
    echo "Running command for image key $image_key and subset $subset clipping from -1000 to 1000 Hu"
    deep-mt scan-stats ./data/configs/imaging/mrs/mrs-ct-ventricles-0.44x0.44x1.0mm.yaml --subset "$subset" --image-key "$image_key" --hu-min -1000 --hu-max 1000
  done
done

# Loop through the image keys and subsets again, but this time without the --hu-min and --hu-max arguments
for image_key in "${image_keys[@]}"; do
  for subset in "${subsets[@]}"; do
    echo "Running command for image key $image_key and subset $subset"
    deep-mt scan-stats ./data/configs/imaging/mrs/mrs-ct-ventricles-0.44x0.44x1.0mm.yaml --subset "$subset" --image-key "$image_key"
  done
done