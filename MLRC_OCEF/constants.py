from recommender_models import ALS, LMF, SVDS

SEED = 42


# appendix C.2, ALS
als_gt_hparams = {
    "factors": [2 ** (i + 4) for i in range(4)],
    "regularization": [10 ** (i - 2) for i in range(4)],
    "alpha": [10 ** (i - 1) for i in range(4)],
}

lmf_gt_hparams = {
    "factors": als_gt_hparams["factors"],
    "regularization": als_gt_hparams["regularization"],
}

# factors from appendix C.2
recommender_hparams_og = {
    "factors": [2**i for i in range(9)],
    # "factors": [4, 6, 8]
    # "regularization": [10 ** (i - 3) for i in range(4)],
}

ground_truth_hparams = {
    ALS: als_gt_hparams,
    LMF: lmf_gt_hparams,
    SVDS: {"factors": als_gt_hparams["factors"]},
}
recommender_hparams = {
    ALS: {
        "factors": recommender_hparams_og["factors"],
        "regularization": [10 ** (i - 3) for i in range(4)],
        "alpha": [10 ** (i - 1) for i in range(4)],
    },
    LMF: {
        "factors": recommender_hparams_og["factors"],
        "regularization": [10 ** (i - 3) for i in range(4)],
    },
    SVDS: recommender_hparams_og,
}


# # testing
# ground_truth_hparams = {
#     ALS: {"factors": [2, 4]},
#     # LMF: lmf_gt_hparams,
#     # SVDS: {"factors": als_gt_hparams["factors"]},
# }
# recommender_hparams = {
#     ALS: {
#         "factors": [2, 5]
#         # "regularization": [10 ** (i - 3) for i in range(4)],
#         # "alpha": [10 ** (i - 1) for i in range(4)],
#     },
#     LMF: {
#         "factors": recommender_hparams_og["factors"],
#         "regularization": [10 ** (i - 3) for i in range(4)],
#     },
#     SVDS: recommender_hparams_og,
# }
