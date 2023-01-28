SEED = 42


# testing
# ground_truth_hparams = {
#     "factors": [4, 8],
#     "regularization": [0.1, 1],
# }

# recommender_hparams = {
#     "factors": [4, 6],
#     "regularization": [0.1, 1],
#     "alpha": [0.01],
# }

# appendix C.2
ground_truth_hparams = {
    "factors": [2 ** (i + 4) for i in range(4)],
    "regularization": [10 ** (i - 2) for i in range(4)],
    "alpha": [10 ** (i - 1) for i in range(4)],
}

# appendix C.2
# recommender_hparams = {
#     "factors": [2**i for i in range(9)],
#     "regularization": [10 ** (i - 3) for i in range(4)],
#     "alpha": [10 ** (i - 1) for i in range(4)],
# }


ground_truth_movielens_hparams = ground_truth_hparams

ground_truth_lastfm_hparams = {
    "factors": ground_truth_movielens_hparams["factors"],
    "regularization": ground_truth_movielens_hparams["regularization"],
    "learning_rate": [0.1, 1.0],
}

recommender_hparams = {
    "n_factors": ground_truth_movielens_hparams["factors"],
    "reg": ground_truth_movielens_hparams["regularization"],
    "lr": [0.005, 0.1],
    "n_epochs": [20, 100],
}
