#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.

library(parallel)
library(optparse)

register_cta_to_ct <-
  function(cta_path,
           ct_pattern = NULL,
           ...) {
    source("registration-functions.R")
    source("img-functions.R")

    print(paste("Register CTA to CT:", cta_path))
    cta_folder = dirname(cta_path)
    print(paste("cta_folder: ", cta_folder))

    # Find CT
    ct_folder = file.path(dirname(cta_folder), "ax_CT")
    print(paste("ct_folder: ", ct_folder))
    print(paste("ct_pattern: ", ct_pattern))
    files = list.files(
      path = ct_folder,
      pattern = ct_pattern,
      recursive = TRUE,
      full.names = TRUE
    )
    
    # Set CT template and transform type
    if (length(files) > 0) {
      ct_path = files[1]
      resample = TRUE
      dilate_mask = TRUE
      voxels = c(1, 1, 2) # Resample image and template to 1.0x1.0x2.0mm
      transform_type = "SyN" # SyNAggro https://antspyx.readthedocs.io/en/latest/registration.html, SyNRA
      print(paste("transform_type: ", transform_type))

      # Strip CTA skull
      # STK1_ax_A_cropped.nii.gz
      # STK1_ax_A_cropped_combined_bet.nii.gz
      cta_file_name = str_replace(basename(cta_path), ".nii.gz", "")
      cta_bet_path = file.path(cta_folder, paste0(cta_file_name, "_combined_bet", ".nii.gz"))
      print(paste("cta_path: ", cta_path))
      print(paste("cta_bet_path: ", cta_bet_path))
      cta_img <- loadSkullStrippedImage(cta_path, cta_bet_path, resample=resample, parameters=voxels, dilate_mask=dilate_mask)

      # Strip CT skull
      # STK1_ax_CT_0.44x0.44x1.0mm_to_scct_unsmooth_SS_0_0.44x0.44x1.0mm_DenseRigid.nii.gz
      # STK1_ax_CT_0.44x0.44x1.0mm_to_scct_unsmooth_SS_0_0.44x0.44x1.0mm_DenseRigid_combined_bet.nii.gz
      ct_file_name = str_replace(basename(ct_path), ".nii.gz", "")
      ct_bet_path = file.path(ct_folder, paste0(ct_file_name, "_combined_bet", ".nii.gz"))
      print(paste("ct_path: ", ct_path))
      print(paste("ct_bet_path: ", ct_bet_path))
      ct_img <- loadSkullStrippedImage(ct_path, ct_bet_path, resample=resample, parameters=voxels, dilate_mask=dilate_mask)

      # Register our brain to the CT brain template
      # Saved to output_path
      output_file_name =  paste0(cta_file_name, "_to_", str_replace(basename(ct_path), ".nii.gz", ""))
      output_path = file.path(cta_folder, paste0(output_file_name, ".nii.gz"))
      print(paste("output_path: ", output_path))

      # Perform transforms
      brain_reg = registration(
        cta_img,
        template.file = ct_img,
        typeofTransform = transform_type,
        interpolator = "Linear"
      )

      # Perform transform on full sized image and template with no skull stripped
      ants_ct <- antsImageRead(ct_path, 3)
      ants_cta <- antsImageRead(cta_path, 3)

      tmp_img <- antsApplyTransforms(
          fixed = ants_ct,
          moving = ants_cta,
          transformlist = brain_reg$fwdtransforms,
          interpolator = "linear"
      )
      antsImageWrite(tmp_img, output_path)

      tmp_img = brain_reg$outfile
      ct_path = ct_img

      # Visualise registration
      print(brain_reg$outfile)
      window = c(0, 100)
      window_brain = window_img(tmp_img, window = window)
      figure_path = file.path(cta_folder, paste0(output_file_name, ".png"))
      print(paste("figure_path: ", figure_path))
      png(figure_path, width = 1000, height = 700)
      double_ortho(ct_path, window_brain, window = window)
      dev.off()

      # Cleanup temp files
      do.call(file.remove, list(list.files(tempdir(), full.names=TRUE)))

      print(paste("Saved registered scan to:", output_path))
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
    c("-a", "--cta-pattern"),
    type = "character",
    default = NULL,
    help = "The CTA pattern to search for, e.g. '.*ax_A_cropped\\.nii\\.gz' will find all cropped CTAs.",
    metavar = "character",
    dest = "cta_pattern"
  ),
  make_option(
    c("-c", "--ct-pattern"),
    type = "character",
    default = NULL,
    help = "The CT pattern to search for, e.g. '.*ax_CT_1\\.0x1\\.0x2\\.0mm_to_scct_unsmooth_1\\.0x1\\.0x2\\.0mm_Rigid\\.nii\\.gz' will find the desired CT scan.",
    metavar = "character",
    dest = "ct_pattern"
  )
)

opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

if (is.null(opt$dir)) {
  print_help(opt_parser)
  stop("Please specify the dir option.", call. = FALSE)
}

if (is.null(opt$cta_pattern)) {
  print_help(opt_parser)
  stop("Please specify the cta-pattern option.", call. = FALSE)
}

if (is.null(opt$ct_pattern)) {
  print_help(opt_parser)
  stop("Please specify the ct-pattern option.", call. = FALSE)
}

# Find all folders matching pattern
files = list.files(
  path = opt$dir,
  pattern = opt$cta_pattern,
  recursive = TRUE,
  full.names = TRUE
)
print(files)

# Run processes in parallel
cl = parallel::makeCluster(detectCores(), outfile = "")
parallel::parLapply(
  cl,
  files,
  register_cta_to_ct,
  ct_pattern = opt$ct_pattern
)
parallel::stopCluster(cl)

