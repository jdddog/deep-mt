#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.

library(parallel)
library(optparse)

register_to_template <-
  function(input_path,
           template_path = NULL,
           transform_type = NULL,
           ...) {
    source("registration-functions.R")
    source("img-functions.R")
    print(input_path)
    print(template_path)
    print(transform_type)

    print(paste("Register to template:", input_path))

    # Load skull stripped image
    file_name = str_replace(basename(input_path), ".nii.gz", "")
    base_folder = dirname(input_path)
    bet_path = file.path(base_folder, paste0(file_name, "_combined_bet", ".nii.gz"))
    img <- loadSkullStrippedImage(input_path, bet_path)

    # Register our brain to the CT brain template
    # Saved to output_path
    template_name  = str_replace(basename(template_path), ".nii.gz", "")
    output_path = file.path(
      base_folder,
      paste0(
        file_name,
        "_to_",
        template_name,
        "_",
        transform_type,
        ".nii.gz"
      )
    )
    figure_path = file.path(base_folder, paste0(file_name, "_to_", template_name, "_", transform_type, ".png"))

    if(!file.exists(output_path) || !file.exists(figure_path)) {
        # Resample image and template to 1.0x1.0x2.0mm
        res_img = resample_image(img, parameters = c(1, 1, 2))
        res_tpl = resample_image(template_path, parameters = c(1, 1, 2))

        # Registration
        brain_reg = registration(
          res_img,
          template.file = res_tpl,
          typeofTransform = transform_type,
          interpolator = "Linear"
        )
        print("Output")
        print(brain_reg)

        # Perform transform on full sized image and template with no skull stripped
        ants_template <- antsImageRead(template_path, 3)
        ants_input <- antsImageRead(input_path, 3)

        tmp_img <- antsApplyTransforms(
            fixed = ants_template,
            moving = ants_input,
            transformlist = brain_reg$fwdtransforms,
            interpolator = "linear"
        )
        antsImageWrite(tmp_img, output_path)

        # Visualise registration
        print(output_path)
        window = c(0, 100)
        window_brain = window_img(output_path, window = window)

        png(figure_path, width = 1000, height = 700)
        double_ortho(template_path, window_brain, window = window)
        dev.off()

        # Cleanup temp files
        do.call(file.remove, list(list.files(tempdir(), full.names=TRUE)))

        print(paste("Saved registered scan to:", output_path))
    } else {
        print(paste("Output already exists:", output_path))
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
    help = "The file pattern to search for, e.g. '.*ax_CT_1\\.0x1\\.0x2\\.0mm\\.nii.gz' will find all CT files ending in that pattern.",
    metavar = "character"
  ),
  make_option(
    c("-t", "--template"),
    type = "character",
    default = NULL,
    help = "The path to the template to use for registration.",
    metavar = "character"
  ),
  make_option(
    c("-r", "--transform"),
    type = "character",
    default = NULL,
    help = "The type of transform to use. See notes typeofTransform for options: https://antspy.readthedocs.io/en/latest/registration.html.",
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

if (is.null(opt$template)) {
  print_help(opt_parser)
  stop("Please specify the template option.", call. = FALSE)
}

if (is.null(opt$transform)) {
  print_help(opt_parser)
  stop("Please specify the transform option.", call. = FALSE)
}

# # Find all folders matching pattern
files = list.files(
  path = opt$dir,
  pattern = opt$pattern,
  recursive = TRUE,
  full.names = TRUE
)
print(files)

# Run processes in parallel
cl = parallel::makeCluster(detectCores(), outfile = "")
parallel::parLapply(
  cl,
  files,
  register_to_template,
  template_path = opt$template,
  transform_type = opt$transform
)
parallel::stopCluster(cl)
