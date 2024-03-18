#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.

library(parallel)
library(optparse)

visualise_image <- function(img_path, ...) {
  library(neurobase)
  library(stringr)
  
  # Constants
  window = c(0, 100)
  img_min_val = -1024
  img_max_val = 3071
  width = 1000
  height = 1000
  
  # Make paths
  base_folder = dirname(img_path)
  file_name = str_replace(basename(img_path), ".nii.gz", "")
  figure_path = file.path(base_folder, paste0(file_name, ".png"))
  
  # Read image
  print(paste("Loading image:", img_path))
  img = readnii(img_path)
  img = rescale_img(img, min.val = img_min_val, max.val = img_max_val)

  # Plot figure
  png(figure_path, width = width, height = height)
  ortho2(img,
         window = window)
  dev.off()
  
  print(paste("Saved visualisation to:", figure_path))
}

# Parse options
option_list = list(
  make_option(
    c("-d", "--dir"),
    type = "character",
    default = NULL,
    help = "The path to the folder containing the .nii.gz files.",
    metavar = "character"
  ),
  make_option(
    c("-p", "--pattern"),
    type = "character",
    default = NULL,
    help = "The nii file name pattern to search for, e.g. '.*ct_bet\\.nii\\.gz' will visualise all CT_BET masks and '.*fsl_bet\\.nii\\.gz' will visualise all FSL BET masks.",
    metavar = "character"
  )
)

opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

if (is.null(opt$dir)) {
  print_help(opt_parser)
  stop("Please specify the dir option.", call. = FALSE)
}

if (is.null(opt$pattern)) {
  print_help(opt_parser)
  stop("Please specify the pattern option.", call. = FALSE)
}

# Find all folders matching pattern
files = list.files(
  path = opt$dir,
  pattern = opt$pattern,
  recursive = TRUE,
  full.names = TRUE
)
print(files)

# Run processes in parallel
cl = parallel::makeCluster(6, outfile="")
parallel::parLapply(cl,
                    files,
                    visualise_image)
parallel::stopCluster(cl)
