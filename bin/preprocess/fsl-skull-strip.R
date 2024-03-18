#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.

library(parallel)
library(optparse)

fsl_skull_strip <- function(input_path, ...) {
  library(neurobase)
  library(ichseg)
  library(gtools)
  library(stringr)
  library(fslr)
  
  # Constants
  window = c(0, 100)
  img_min_val = -1024
  img_max_val = 3071
  
  print(paste("Running FSL BET on:", input_path))
  
  # Make paths
  base_folder = dirname(input_path)
  file_name = str_replace(basename(input_path), ".nii.gz", "")
  
  # Read image
  img = readnii(input_path)
  img = rescale_img(img, min.val = img_min_val, max.val = img_max_val)
  
  # Skull strip image
  mask_path = file.path(base_folder, paste0(file_name, "_fsl_bet", ".nii.gz"))
  CT_Skull_Strip(
    img,
    maskfile = mask_path,
    opts = "-f 0.01 -v",
    lthresh = -20,
    sigma = 3,
    verbose = TRUE
  )

  # Cleanup temp files
  do.call(file.remove, list(list.files(tempdir(), full.names=TRUE)))
  
  print(paste("Saved FSL BET to:", mask_path))
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
    help = "The brain mask file pattern to search for, e.g. '.*ct_bet\\.nii\\.gz' will visualise all CT_BET masks and '.*fsl_bet\\.nii\\.gz' will visualise all FSL BET masks.",
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
cl = parallel::makeCluster(detectCores(), outfile = "")
parallel::parLapply(cl,
                    files,
                    fsl_skull_strip)
parallel::stopCluster(cl)
