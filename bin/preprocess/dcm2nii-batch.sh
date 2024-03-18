#!/usr/bin/env bash

#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.

# Parse arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 INPUT_DIR OUTPUT_DIR SCAN_TYPE"
    echo
    echo "Convert DICOM files into NIfTI files with dcm2niix."
    echo
    echo "Arguments:"
    echo "  INPUT_DIR    The path to the DICOM files, e.g. /path/to/dicoms."
    echo "  OUTPUT_DIR   The path where the NIfTI files will be saved, e.g. /path/to/nii."
    echo "  SCAN_TYPE    The type of scan, either ax_CT or ax_A."
    echo
    echo "Options:"
    echo "  -h, --help    Display this help message and exit."
    echo
    exit 1
fi
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
fi
if [ "$#" -ne 3 ]; then
    echo "Error: three arguments are required." >&2
    usage
fi
# Validate SCAN_TYPE
if [ "$3" != "ax_CT" ] && [ "$3" != "ax_A" ]; then
    echo "Error: SCAN_TYPE argument must be 'ax_CT' or 'ax_A'." >&2
    usage
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
SCAN_TYPE=$3
echo "INPUT_DIR: $INPUT_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "SCAN_TYPE: $SCAN_TYPE"

# Get the directory of the current script
# Navigate to the script's directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# Download dcm2niix if not present
file="dcm2niix_lnx.zip"
url="https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20211006/dcm2niix_lnx.zip"

if [ ! -f "$file" ]; then
    wget "$url" -O "$file" && unzip "$file"
fi

# Find scans
pattern=".*\($SCAN_TYPE\)$"
dcm_paths=$(find ${INPUT_DIR} -type d -regex $pattern) # both CT and CTA '.*\(ax_CT\|ax_A\)$'

for path in ${dcm_paths}
do
  IFS='/' read -r -a parts <<< "${path}"
  centre=${parts[-3]}
  case_id=${parts[-2]}
  scan_type=${parts[-1]}
  output=${OUTPUT_DIR}/${centre}/${case_id}/${scan_type}
  mkdir -p ${output}
  file_name=${case_id}_${scan_type}
  echo "./dcm2niix -z y -f ${file_name} -m y -o ${output} ${path}"
  ./dcm2niix -z y -f ${file_name} -m y -o ${output} ${path}
done


