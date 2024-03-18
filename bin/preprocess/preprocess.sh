#!/usr/bin/env bash

#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.


export CUDA_VISIBLE_DEVICES=0

# Parse keyword arguments
# TODO: fail if not all arguments provided
# TODO: provide help message

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -t|--template-folder)
      template_folder="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--nii-path)
      nii_path="$2"
      shift # past argument
      shift # past value
      ;;
    -d|--deep-skull-path)
      deep_skull_path="$2"
      shift # past argument
      shift # past value
      ;;
    -x|--voxel-x)
      voxel_x="$2"
      shift # past argument
      shift # past value
      ;;
    -y|--voxel-y)
      voxel_y="$2"
      shift # past argument
      shift # past value
      ;;
    -z|--voxel-z)
      voxel_z="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      shift # past argument
      ;;
  esac
done

echo "--template-folder " ${template_folder}
echo "--nii-path " ${nii_path}
echo "--deep-skull-path " ${deep_skull_path}
echo "--voxel-x " ${voxel_x}
echo "--voxel-y " ${voxel_y}
echo "--voxel-z " ${voxel_z}

# Voxel sizes escaped for R regexes
voxel_x_es=$(echo ${voxel_x} | sed 's/\./\\\./g')
voxel_y_es=$(echo ${voxel_y} | sed 's/\./\\\./g')
vozel_z_es=$(echo ${voxel_z} | sed 's/\./\\\./g')

# Source deep skull venv
source ${deep_skull_path}/venv/bin/activate

# The CT pattern and template name for Python
ct_pattern=ax_CT_${voxel_x}x${voxel_y}x${voxel_z}mm
template_name=scct_unsmooth_SS_0_${voxel_x}x${voxel_y}x${voxel_z}mm

# The escaped ct pattern and template name
ct_pattern_es=ax_CT_${voxel_x_es}x${voxel_y_es}x${vozel_z_es}mm
template_name_es=scct_unsmooth_SS_0_${voxel_x_es}x${voxel_y_es}x${vozel_z_es}mm

# The template to use
template_path=${template_folder}/${template_name}.nii.gz

# Resample templates
echo "Resampling templates..."
deep-skull resample ${template_folder} scct_unsmooth.nii.gz --x ${voxel_x} --y ${voxel_y} --z ${voxel_z}
deep-skull resample ${template_folder} scct_unsmooth_SS_0.01.nii.gz --x ${voxel_x} --y ${voxel_y} --z ${voxel_z}

# Resample images
echo "Resampling images..."
# TODO: don't resample if files already exist
deep-skull resample ${nii_path} ax_CT.nii.gz --x ${voxel_x} --y ${voxel_y} --z ${voxel_z}

# Strip skulls
echo "Stripping skulls..."
Rscript --vanilla fsl-skull-strip.R -d ${nii_path} -p .*${ct_pattern_es}\\.nii\\.gz
deep-skull extract-brain ${nii_path} ${ct_pattern}.nii.gz --num-workers 1
deep-skull combine-masks ${nii_path} ${ct_pattern}.nii.gz

# Transform CTs to Template
echo "Transform CTs to Template..."
Rscript --vanilla reg-to-template.R -d ${nii_path} -p .*${ct_pattern_es}\\.nii\\.gz -t ${template_path} -r DenseRigid

# Strip Skulls for Transformed CTs
echo "Strip Skulls for Transformed CTs..."
Rscript --vanilla fsl-skull-strip.R -d ${nii_path} -p .*${ct_pattern_es}_to_${template_name_es}_DenseRigid\\.nii\\.gz
deep-skull extract-brain ${nii_path} ${ct_pattern}_to_${template_name}_DenseRigid.nii.gz --num-workers 1 --overwrite
deep-skull combine-masks ${nii_path} ${ct_pattern}_to_${template_name}_DenseRigid.nii.gz
Rscript --vanilla vis-brain-masks.R -d ${nii_path} -p "^STK(CH)?[0-9_]+${ct_pattern_es}_to_${template_name_es}_DenseRigid\\.nii\\.gz$"

# Extract skull of CTA
cta_pattern=ax_A_cropped
# TODO: don't repeat this step if running command for a different size and images already exist
Rscript --vanilla fsl-skull-strip.R -d ${nii_path} -p .*${cta_pattern}\\.nii\\.gz
deep-skull extract-brain ${nii_path} ${cta_pattern}.nii.gz --num-workers 1
deep-skull combine-masks ${nii_path} ${cta_pattern}.nii.gz

# Transform CTAs to CT
echo "Transform CTAs to CT..."
Rscript --vanilla reg-cta-to-ct.R --dir=${nii_path} --cta-pattern=.*ax_A_cropped\\.nii\\.gz --ct-pattern=.*${ct_pattern_es}_to_${template_name_es}_DenseRigid\\.nii\\.gz
