SEED = 42

# appendix C.2
ground_truth_hparams = {
    "factors": [2 ** (i + 4) for i in range(4)],
    "regularization": [10 ** (i - 2) for i in range(4)],
    "alpha": [10 ** (i - 1) for i in range(4)],
}

# appendix C.2
recommender_hparams = {
    "factors": [2**i for i in range(9)],
    "regularization": [10 ** (i - 3) for i in range(4)],
    "alpha": [10 ** (i - 1) for i in range(4)],
}

# ground_truth_hparams = {
#     "factors": [4, 8],
#     "regularization": [0.1, 1],
#     "alpha": [0.01],
# }

# recommender_hparams = {
#     "factors": [4, 6],
#     "regularization": [0.1, 1],
#     "alpha": [0.01],
# }
