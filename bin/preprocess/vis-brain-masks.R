#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.

library(parallel)
library(optparse)


visualise_masks <- function(image_path, ...) {
  library(stringr)
  library(fs)
  source("visualise-mask.R")

  # Make paths
  base_folder = dirname(image_path)
  file_name = str_replace(basename(image_path), ".nii.gz", "")
  ct_bet_path = file.path(base_folder, paste0(file_name, "_ct_bet.nii.gz"))
  fsl_bet_path = file.path(base_folder, paste0(file_name, "_fsl_bet.nii.gz"))
  combined_bet_path = file.path(base_folder, paste0(file_name, "_combined_bet.nii.gz"))
  print("Mask paths")
  print(ct_bet_path)
  print(fsl_bet_path)
  print(fsl_bet_path)

  # Visualise CT_BET mask
  if(is_file(ct_bet_path)) {
    visualise_mask(image_path, ct_bet_path)
  }

  # Visualise FSL BET mask
  if(is_file(fsl_bet_path)) {
    visualise_mask(image_path, fsl_bet_path)
  }

  # Visualise combined bet mask
  if(is_file(combined_bet_path)) {
    visualise_mask(image_path, combined_bet_path)
  }
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
    help = "The file pattern to search for.",
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
cl = parallel::makeCluster(detectCores(), outfile="")
parallel::parLapply(cl,
                    files,
                    visualise_masks)
parallel::stopCluster(cl)
