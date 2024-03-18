#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.
#
# Source: https://github.com/muschellij2/extrantsr/blob/5fafdbefa069d3ba869f104d1e0d8c0153281ad7/R/ants_regwrite2.R

library(neurobase)
library(ichseg)
library(gtools)
library(fslr)
library(stringr)
library(ANTsRCore)
library(extrantsr)

setGeneric(
  name = ".int_antsExtractXptrAsString",
  def = function(object)
    standardGeneric(".int_antsExtractXptrAsString")
)


#' @aliases .int_antsExtractXptrAsString,antsImage-method
setMethod(
  f = ".int_antsExtractXptrAsString",
  signature = c("antsImage"),
  definition = function(object) {
    return(as.character(c(object@pointer)))
  }
)

#' @aliases .int_antsExtractXptrAsString,antsMatrix-method
setMethod(
  f = ".int_antsExtractXptrAsString",
  signature = c("antsMatrix"),
  definition = function(object) {
    return(as.character(c(object@pointer)))
  }
)

#' @aliases .int_antsExtractXptrAsString,numeric-method
setMethod(
  f = ".int_antsExtractXptrAsString",
  signature = c("numeric"),
  definition = function(object) {
    return(object)
  }
)

#' @aliases .int_antsExtractXptrAsString,character-method
setMethod(
  f = ".int_antsExtractXptrAsString",
  signature = c("character"),
  definition = function(object) {
    return(object)
  }
)

.int_antsProcessArguments <- function(args) {
  char_vect <- ""
  if (typeof(args) == "list") {
    char_vect <- NULL
    for (i in (1:length(args))) {
      if (length(names(args)) != 0) {
        if (nchar(names(args)[i]) > 1) {
          char_vect <- c(char_vect, paste("--", names(args)[i], sep = ""))
        } else {
          char_vect <- c(char_vect, paste("-", names(args)[i], sep = ""))
        }
      }
      if (typeof(args[[i]]) == "list") {
        char_vect <- c(char_vect, paste(args[[i]]$name, "[", sep = ""))
        args[[i]]$name <- NULL
        for (j in (1:length(args[[i]]))) {
          char_vect <-
            c(char_vect,
              as.character(.int_antsExtractXptrAsString(args[[i]][[j]])))
        }
        char_vect <- c(char_vect, "]")
      } else {
        char_vect <-
          c(char_vect,
            as.character(.int_antsExtractXptrAsString(args[[i]])))
      }
    }
  }
  return(char_vect)
}

antsApplyTransforms <- function(fixed,
                                moving,
                                transformlist = "",
                                interpolator = c(
                                  "linear",
                                  "nearestNeighbor",
                                  "multiLabel",
                                  "gaussian",
                                  "bSpline",
                                  "cosineWindowedSinc",
                                  "welchWindowedSinc",
                                  "hammingWindowedSinc",
                                  "lanczosWindowedSinc",
                                  "genericLabel"
                                ),
                                imagetype = 0,
                                whichtoinvert = NA,
                                compose = NA,
                                verbose = FALSE,
                                ...) {
  if (is.character(fixed)) {
    if (fixed == "-h") {
      .Call("antsApplyTransforms",
            .int_antsProcessArguments(list("-h")),
            PACKAGE = "ANTsRCore")
      return()
    }
  }
  if (missing(fixed) |
      missing(moving) | missing(transformlist)) {
    print("missing inputs")
    return(NA)
  }
  interpolator[1] = paste(
    tolower(substring(interpolator[1], 1, 1)),
    substring(interpolator[1], 2),
    sep = "",
    collapse = " "
  )
  interpOpts = c(
    "linear",
    "nearestNeighbor",
    "multiLabel",
    "gaussian",
    "bSpline",
    "cosineWindowedSinc",
    "welchWindowedSinc",
    "hammingWindowedSinc",
    "lanczosWindowedSinc",
    "genericLabel"
  )
  interpolator <- match.arg(interpolator, interpOpts)
  
  if (is.antsImage(transformlist)) {
    warning("transformlist is an antsImage, creating a temporary file")
    tfile = tempfile(fileext = ".nii.gz")
    antsImageWrite(transformlist, filename = tfile)
    transformlist = tfile
  }
  
  if (!is.character(transformlist)) {
    warning("transformlist is not a character vector")
  }
  
  args <- list(fixed, moving, transformlist, interpolator, ...)
  if (!is.character(fixed)) {
    moving = check_ants(moving)
    if (fixed@class[[1]] == "antsImage" &
        moving@class[[1]] == "antsImage") {
      for (i in 1:length(transformlist)) {
        if (!file.exists(transformlist[i])) {
          stop(paste("Transform ",
                     transformlist[i],
                     " does not exist.",
                     sep = ""))
        }
      }
      inpixeltype <- fixed@pixeltype
      warpedmovout <- antsImageClone(moving)
      f <- fixed
      m <- moving
      if ((moving@dimension == 4) & (fixed@dimension == 3) &
          (imagetype == 0))
        stop("Set imagetype 3 to transform time series images.")
      mytx <- list()
      # If whichtoinvert is NA, then attempt to guess the right thing to do
      #
      # If the transform list is (affine.mat, warp), whichtoinvert = c("T", "F")
      #
      # else whichtoinvert = rep("F", length(transformlist))
      if (all(is.na(whichtoinvert))) {
        if (length(transformlist) == 2 &
            grepl("\\.mat$", transformlist[1]) &
            !(grepl("\\.mat$", transformlist[2]))) {
          whichtoinvert <- c(TRUE, FALSE)
        }
        else {
          whichtoinvert <- rep(FALSE, length(transformlist))
        }
      }
      if (length(whichtoinvert) != length(transformlist)) {
        stop("Transform list and inversion list must be the same length")
      }
      for (i in c(1:length(transformlist))) {
        ismat <- FALSE
        if (grepl("\\.mat$", transformlist[i]) ||
            grepl("\\.txt$", transformlist[i])) {
          ismat <- TRUE
        }
        if (whichtoinvert[i] && !(ismat)) {
          # Can't invert a warp field, user should pass inverseWarp directly. Something wrong
          stop(
            paste(
              "Cannot invert transform " ,
              i ,
              " ( " ,
              transformlist[i],
              " ), because it is not a matrix. ",
              sep = ""
            )
          )
        }
        if (whichtoinvert[i]) {
          mytx <- list(mytx,
                       "-t",
                       paste("[", transformlist[i], ",1]",
                             sep = ""))
        }
        else {
          mytx <- list(mytx, "-t", transformlist[i])
        }
        
      }
      if (is.na(compose))
        args <-
        list(
          d = fixed@dimension,
          i = m,
          o = warpedmovout,
          r = f,
          n = interpolator,
          unlist(mytx)
        )
      tfn <- paste(compose, "comptx.nii.gz", sep = '')
      if (!is.na(compose)) {
        mycompo = paste("[", tfn, ",1]", sep = "")
        args <-
          list(
            d = fixed@dimension,
            i = m,
            o = mycompo,
            r = f,
            n = interpolator,
            unlist(mytx)
          )
      }
      print(paste0("MY ARGS: "))
      myargs <- .int_antsProcessArguments(c(args))
      for (jj in c(1:length(myargs))) {
        if (!is.na(myargs[jj])) {
          if (myargs[jj] == "-") {
            myargs2 <- rep(NA, (length(myargs) - 1))
            myargs2[1:(jj - 1)] <- myargs[1:(jj - 1)]
            myargs2[jj:(length(myargs) - 1)] <-
              myargs[(jj + 1):(length(myargs))]
            myargs <- myargs2
          }
        }
      }
      myverb = as.numeric(verbose)
      if (verbose)
        print(myargs)
      .Call(
        "antsApplyTransforms",
        c(
          myargs,
          "-z",
          1,
          "-v",
          myverb,
          "--float",
          1,
          "-e",
          imagetype,
          "-f",
          -1024
        ),
        PACKAGE = "ANTsRCore"
      )
      if (is.na(compose))
        return(antsImageClone(warpedmovout, inpixeltype))
      if (!is.na(compose))
        if (file.exists(tfn))
          return(tfn)
      else
        return(NA)
    }
    # Get here if fixed, moving, transformlist are not missing, fixed is not of type character,
    # and fixed and moving are not both of type antsImage
    stop(
      paste0(
        'fixed, moving, transformlist are not missing,',
        ' fixed is not of type character,',
        ' and fixed and moving are not both of type antsImage'
      )
    )
    return(1)
  }
  # if ( Sys.info()['sysname'] == 'XXX' ) { mycmd<-.antsrParseListToString( c(args)
  # ) system( paste('antsApplyTransforms ', mycmd$mystr ) ) return( antsImageRead(
  # mycmd$outimg, as.numeric(mycmd$outdim) ) ) }
  .Call("antsApplyTransforms",
        .int_antsProcessArguments(c(
          args, "-z", 1, "--float", 1, "-e", imagetype, "-f",-1024
        )),
        PACKAGE = "ANTsRCore")
}

registration <- function(filename,
                         skull_strip = FALSE,
                         correct = FALSE,
                         correction = "N4",
                         retimg = TRUE,
                         outfile = NULL,
                         template.file = file.path(fsldir(), "data",
                                                   "standard",
                                                   "MNI152_T1_1mm_brain.nii.gz"),
                         interpolator = "Linear",
                         other_interpolator = interpolator,
                         other.files = NULL,
                         other.outfiles = NULL,
                         other.init = NULL,
                         invert_interpolator = interpolator,
                         invert.native.fname = NULL,
                         invert.file = NULL,
                         typeofTransform = "SyN",
                         remove.warp = FALSE,
                         outprefix = NULL,
                         bet.opts = "-B -f 0.1 -v",
                         betcmd = "bet",
                         copy_origin = TRUE,
                         verbose = TRUE,
                         reproducible = TRUE,
                         seed = 1,
                         force_registration = TRUE,
                         ...) {
  if (reproducible) {
    if (is.null(seed)) {
      stop("For reproducible = TRUE, you must set a seed!")
    }
    Sys.setenv(ANTS_RANDOM_SEED = seed)
    itk_threads = Sys.getenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")
    Sys.setenv(ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = 1)
    on.exit({
      Sys.setenv(ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = itk_threads)
    })
  }
  outfile = check_outfile(outfile = outfile, retimg = retimg)
  
  have.other = FALSE
  if (!is.null(other.files)) {
    have.other = TRUE
    other.files = checkimg(other.files)
    if (!is.null(other.outfiles)) {
      other.outfiles = checkimg(other.outfiles)
    } else {
      other.outfiles = sapply(other.outfiles, function(x) {
        tempfile(fileext = ".nii.gz")
      })
    }
    
    lother = length(other.files)
    lout = length(other.outfiles)
    if (lother != lout) {
      stop("Other outfile and infiles must be same length")
    }
    if (!is.null(other.init)) {
      ltrans = length(other.init)
      if (ltrans != lout) {
        stop("Other initial transformations and infiles must be same length")
      }
    }
  }
  
  if (!is.null(invert.file)) {
    invert.file = checkimg(invert.file)
    if (is.null(invert.native.fname)) {
      invert.native.fname = sapply(seq_along(invert.file), function(x) {
        tempfile(fileext = ".nii.gz")
      })
    }
    
    if (length(invert.file) != length(invert.native.fname)) {
      stop("Length of invert.file and invert.native.fnames must be equal!")
    }
  }
  
  # if (!remove.warp) {
  # stopifnot(!is.null(outprefix))
  # } else {
  if (is.null(outprefix)) {
    outprefix = tempfile()
  }
  dir.create(dirname(outprefix),
             showWarnings = FALSE,
             recursive = TRUE)
  # }
  
  if (ANTsRCore::is.antsImage(filename)) {
    t1 = antsImageClone(filename)
  } else {
    filename = checkimg(filename)
    # 	filename = path.expand(filename)
    stopifnot(file.exists(filename))
    t1 <- antsImageRead(filename, 3)
  }
  
  if (skull_strip) {
    if (verbose) {
      message("# Skull Stripping\n")
    }
    if (ANTsRCore::is.antsImage(filename)) {
      filename = checkimg(filename)
    }
    ext = get.imgext()
    bet_file = tempfile()
    x = fslbet(
      infile = filename,
      outfile = bet_file,
      opts = bet.opts,
      betcmd = betcmd,
      retimg = FALSE
    )
    bet_file = paste0(tempfile(), ext)
    bet_maskfile = paste0(tempfile(), "_Mask", ext)
    # bet = antsImageRead(bet_file, 3)
    bet_mask = antsImageRead(bet_maskfile, 3)
  }
  
  t1N3 <- antsImageClone(t1)
  
  if (have.other) {
    stopifnot(all(file.exists(other.files)))
    other.imgs = lapply(other.files, antsImageRead,
                        dimension = 3)
    if (copy_origin) {
      other.imgs = lapply(other.imgs,
                          antsCopyOrigin,
                          reference = t1N3)
    }
    N3.oimgs = lapply(other.imgs, antsImageClone)
  }
  ##
  if (correct) {
    if (verbose) {
      message("# Running Bias-Field Correction on file\n")
    }
    t1N3 = bias_correct(
      file = t1,
      correction = correction,
      retimg = TRUE,
      verbose = verbose
    )
    t1N3 = oro2ants(t1N3)
    if (have.other) {
      if (verbose) {
        message("# Running Bias-Field Correction on other.files\n")
      }
      for (i in seq(lother)) {
        N3.oimgs[[i]] = bias_correct(
          file = other.imgs[[i]],
          correction = correction,
          retimg = TRUE,
          verbose = verbose
        )
        N3.oimgs[[i]] = oro2ants(N3.oimgs[[i]])
      }
    }
  }
  
  if (skull_strip) {
    t1N3 = maskImage(t1N3, bet_mask)
    if (have.other) {
      N3.oimgs = lapply(N3.oimgs, maskImage,
                        img.mask = bet_mask)
    }
    rm(list = "bet_mask")
    gc()
    
  }
  
  ##
  if (ANTsRCore::is.antsImage(template.file)) {
    template = antsImageClone(template.file)
  } else {
    template.file = checkimg(template.file)
    stopifnot(file.exists(template.file))
    
    template.file = path.expand(template.file)
    template <- antsImageRead(template.file)
  }
  # template.img <- readnii(template.path, reorient = FALSE)
  
  if (verbose) {
    message("# Running Registration of file to template\n")
  }
  out_trans = c(Affine = "0GenericAffine.mat",
                fwd = "1Warp.nii.gz",
                inv = "1InverseWarp.nii.gz")
  n_trans = names(out_trans)
  out_trans = paste0(outprefix, out_trans)
  names(out_trans) = n_trans
  
  if (!all(file.exists(out_trans)) || force_registration) {
    antsRegOut.nonlin <- ANTsRCore::antsRegistration(
      fixed = template,
      moving = t1N3,
      typeofTransform = typeofTransform,
      outprefix = outprefix,
      verbose = verbose,
      ...
    )
  } else {
    antsRegOut.nonlin = list(fwdtransforms = unname(out_trans[c("fwd", "Affine")]),
                             invtransforms = unname(out_trans[c("Affine", "inv")]))
  }
  ######################################################
  # added this to try to wrap up the gc()
  antsRegOut.nonlin$warpedmovout = NULL
  antsRegOut.nonlin$warpedfixout = NULL
  ######################################################
  for (i in 1:5) {
    gc()
  }
  # fixing multi-naming convention problem
  fwd = antsRegOut.nonlin$fwdtransforms
  fwd = fwd[grepl("Generic|Warp", fwd)]
  antsRegOut.nonlin$fwdtransforms = fwd
  
  inv = antsRegOut.nonlin$invtransforms
  inv = inv[grepl("Generic|Warp", inv)]
  antsRegOut.nonlin$invtransforms = inv
  
  
  if (verbose) {
    message("# Applying Registration output is\n")
    print(antsRegOut.nonlin)
  }
  
  if (!all(file.exists(antsRegOut.nonlin$fwdtransforms))) {
    stop("ANTs Registration did not complete, transforms do not exist!")
  }
  if (!all(file.exists(antsRegOut.nonlin$invtransforms))) {
    stop("ANTs Registration did not complete, inverse transforms do not exist!")
  }
  
  if (verbose) {
    message("# Applying Transformations to file\n")
    #     message("# Fixed is \n")
    #     print(template)
    #     message("# Moving is \n")
    #     print(t1N3)
  }
  t1.to.template <- antsApplyTransforms(
    fixed = template,
    moving = t1N3,
    transformlist = antsRegOut.nonlin$fwdtransforms,
    interpolator = interpolator,
    verbose = verbose
  )
  # "-h" works
  
  # moving = t1N3
  transformlist = antsRegOut.nonlin$invtransforms
  # dimension = 3
  
  output = paste0(tempfile(), ".nii.gz")
  
  if (!is.null(invert.file)) {
    if (verbose) {
      message("# Applying Inverse transforms to invert.file\n")
    }
    for (iatlas in seq_along(invert.file)) {
      output = invert.native.fname[iatlas]
      
      atlas = antsImageRead(invert.file[iatlas])
      # 			if (!grepl("[.]nii$|[.]nii[.]gz$", output)) {
      # 				output = paste0(output, ".nii.gz")
      # 			}
      
      tmp_img = antsApplyTransforms(
        fixed = t1N3,
        moving = atlas,
        transformlist = transformlist,
        interpolator = invert_interpolator,
        verbose = verbose
      )
      antsImageWrite(tmp_img, output)
      rm(list = c("tmp_img", "atlas"))
      gc()
      
    }
  }
  
  
  
  if (have.other) {
    if (verbose) {
      message("# Applying Transforms to other.files\n")
    }
    if (is.null(other.init)) {
      reg.oimgs = lapply(N3.oimgs, function(x) {
        antsApplyTransforms(
          fixed = template,
          moving = x,
          transformlist = antsRegOut.nonlin$fwdtransforms,
          interpolator = other_interpolator,
          verbose = verbose
        )
      })
    } else {
      reg.oimgs = mapply(function(x, y) {
        antsApplyTransforms(
          fixed = template,
          moving = x,
          transformlist = c(antsRegOut.nonlin$fwdtransforms, y),
          interpolator = other_interpolator,
          verbose = verbose
        )
      }, N3.oimgs, other.init, SIMPLIFY = FALSE)
    }
  }
  
  if (verbose) {
    message("# Writing out file\n")
  }
  antsImageWrite(t1.to.template, outfile)
  if (verbose) {
    print(outfile)
  }
  rm(list = "t1.to.template")
  gc()
  
  if (have.other) {
    if (verbose) {
      message("# Writing out other.files\n")
    }
    for (i in seq(lother)) {
      antsImageWrite(reg.oimgs[[i]],
                     other.outfiles[i])
      if (verbose) {
        print(other.outfiles[i])
      }
    }
    rm(list = c("reg.oimgs", "N3.oimgs"))
    gc()
    
  }
  
  if (remove.warp) {
    if (verbose) {
      message("# Removing Warping images\n")
    }
    files = unlist(antsRegOut.nonlin[c("fwdtransforms", "invtransforms")])
    files = grep("Warp", files, value = TRUE)
    if (length(files) > 0) {
      file.remove(files)
    }
  }
  if (retimg) {
    if (verbose) {
      message("# Reading data back into R\n")
    }
    outfile = readnii(outfile, reorient = FALSE)
  }
  
  L = list(
    outfile = outfile,
    fwdtransforms = antsRegOut.nonlin$fwdtransforms,
    invtransforms = antsRegOut.nonlin$invtransforms,
    interpolator = interpolator,
    other_interpolator = other_interpolator,
    invert_interpolator = invert_interpolator,
    typeofTransform = typeofTransform,
    retimg = retimg
  )
  L$inverted.outfiles = invert.native.fname
  
  L$other.outfiles = other.outfiles
  rm(list = c("t1", "t1N3", "template"))
  gc()
  
  rm(list = "antsRegOut.nonlin")
  for (i in 1:5) {
    gc()
  }
  return(L)
}