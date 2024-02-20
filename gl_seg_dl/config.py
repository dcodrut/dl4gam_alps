# ######################################################################################################################
# ######################################################## S2 ##########################################################
# ######################################################################################################################
class S1:
    DIR_RAW_DATA = f'../data/external/Data4ML_v3/MtBlanc/'
    WD = '../data/external/wd'  # working directory
    DIR_AUX_DATA = f'{WD}/aux_data'
    DIR_OUTLINES_SPLIT = f'{WD}/outlines_split'
    NUM_CV_FOLDS = 5
    VALID_FRACTION = 0.15
    DIR_NC_RASTERS = f'{WD}/rasters/v3'
    # DIR_NC_RASTERS = f'{DIR}/rasters_cropped_test_cv_1'
    MIN_GLACIER_AREA = 0.1  # km2
    DIRS_INFER = [DIR_NC_RASTERS]  # directories on which to make glacier-wide inferences
    NUM_CORES_PREP_DATA = 32
    NUM_CORES_EVAL = 16

    # patch sampling settings
    PATCH_RADIUS = 64
    SAMPLING_STEP = 32
    DIR_GL_PATCHES = f'{WD}/patches_orig_r_{PATCH_RADIUS}_s_{SAMPLING_STEP}'
