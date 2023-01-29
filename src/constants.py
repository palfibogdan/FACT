SEED = 42

# appendix C.2
ground_truth_hparams = {
    "factors": [2 ** (i + 4) for i in range(4)],
    "regularization": [10 ** (i - 2) for i in range(4)],
    "alpha": [10 ** (i - 1) for i in range(4)],
}

ground_truth_lastfm_hparams = {
    "factors": ground_truth_hparams["factors"],
    "regularization": ground_truth_hparams["regularization"],
    "learning_rate": [0.1, 1.0],
}


# appendix C.2
# recommender_hparams = {
#     "factors": [2**i for i in range(9)],
#     "regularization": [10 ** (i - 3) for i in range(4)],
#     "alpha": [10 ** (i - 1) for i in range(4)],
# }
recommender_hparams = {
    "factors": ground_truth_hparams["factors"],
    "regularization": ground_truth_hparams["regularization"],
    "lr": [0.005, 0.1],
}


# testing
# ground_truth_hparams = {
#     "factors": [4, 8],
#     "regularization": [0.1, 1],
# }

# recommender_hparams = {
#     "factors": [4, 6],
#     "regularization": [0.1, 1],
# }


GROUND_TRUTH_HP = {
    "movielens": ground_truth_hparams,
    "lastfm": ground_truth_lastfm_hparams,
    # "lastfm": ground_truth_hparams,
}
