#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.


loadSkullStrippedImage <- function(img_path,
                                   bet_path,
                                   img_min_val = -1024,
                                   img_max_val = 3071,
                                   resample = FALSE,
                                   parameters = c(1, 1, 2),
                                   dilate_mask = FALSE,
                                   ...) {
    # Read image
    img = readnii(img_path)
    img = rescale_img(img, min.val = img_min_val, max.val = img_max_val)

    # Load brain mask and mask brain image
    bet_mask = readnii(bet_path)

    # Resample
    if(resample) {
        img = extrantsr::resample_image(img, parameters = parameters)
        bet_mask = resample_image(bet_mask, parameters = parameters)
    }

    # Dilate mask
    if(dilate_mask) {
        bet_mask = fslr::fsl_dilate(bet_mask, kopts = "-kernel boxv 5", retimg=TRUE)
    }

    # Apply mask to image
    bet_mask[bet_mask == 0] = NA
    img = mask_img(img, bet_mask, allow.NA = TRUE)
    img[is.na(bet_mask)] = img_min_val

    return(img)
}