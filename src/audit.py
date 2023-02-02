import numpy as np


def audit(M, alpha, delta=0.05, epsilon=0.05, gamma=0.1, lmbda=0.1):

    # draw a sample of log(3/delta)/lmbda from M
    users = np.random.choice(M.keys(), int(np.log(3 / delta) / lmbda))

    for user in users:
        # sample log(3*len(M)/delta)/log(1/(1-gamma)) from M, without user
        arms = np.random.choice(
            M.keys()[M.keys() != user],
            int(np.log(3 * len(M) / delta) / np.log(1 / (1 - gamma))),
        )

        arms = np.concatenate((user, arms))

        S = {arm: M[arm] for arm in arms}
        print(S)


M = {
    "0": 0.6,
    "1": 0.3,
    "2": 0.3,
    "3": 0.3,
    "4": 0.3,
    "5": 0.3,
    "6": 0.3,
    "7": 0.3,
    "8": 0.3,
    "9": 0.3,
}
# audit(M)
