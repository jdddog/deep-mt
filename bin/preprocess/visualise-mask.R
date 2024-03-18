#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.

visualise_mask <- function(image_path, mask_path, ...) {
  library(neurobase)

  # Constants
  window = c(0, 100)
  img_min_val = -1024
  img_max_val = 3071
  width = 1000
  height = 1000

  # Make figure path
  base_folder = dirname(image_path)
  figure_path = file.path(base_folder, paste0(str_replace(basename(mask_path), ".nii.gz", ""), ".png"))

  # Read image
  print(paste("Loading image:", image_path))
  img = readnii(image_path)
  img = rescale_img(img, min.val = img_min_val, max.val = img_max_val)

  # Read mask
  print(paste("Loading mask:", mask_path))
  mask = readnii(mask_path)

  # Plot figure
  png(figure_path, width = width, height = height)
  ortho2(img,
         mask,
         window = window,
         col.y = scales::alpha("red", 0.5))
  dev.off()

  print(paste("Saved brain mask visualisation to:", figure_path))
}