import random
import numpy as np

# TODO:
# Implement conservative constraint Xi
# Implement observe, action and reward part
# Verify indices for Beta (Lemma 4 returns Beta_t but OCEF uses Beta_(t-1) )
# Implement get_phi from Lemma 5 and 6


def update_bounds(delta, N, rewards):

    # where is omega specified?
    sigma = 0.5
    omega = 0.5  # random in (0,1)

    theta = np.log(1 + omega) * ((omega * delta) / (2 * (2 + omega))) ** (
        1 / (1 + omega)
    )

    betas = []
    low_bounds = []
    high_bounds = []
    K = len(N)
    for n, r in zip(N, rewards):
        if n == 0:
            mean_mu = 0
        else:
            mean_mu = r / n

        beta = np.sqrt(
            (2 * sigma**2 * (1 + np.sqrt(omega)) ** 2 * (1 + omega)) / N
        ) * np.sqrt(np.log(2 * (K + 1) / theta * np.log((1 + omega) * N)))
        betas.append(beta)

        low_bounds.append(mean_mu - beta)
        high_bounds.append(mean_mu + beta)

    return beta, low_bounds, high_bounds


def remove_non_envy_elements(S, low_bounds, high_bounds, epsilon):
    for k in S:
        if high_bounds[k] <= low_bounds[0] + epsilon:
            S.remove(k)
    return S


def exists_higher_utility(S, low_bounds, high_bounds):
    for k in S:
        if low_bounds[k] > high_bounds[0]:
            return True
    return False


def get_conservarive_constraint(
    t, A, rewards, Phi, low_bound_l, high_bound_0, N, alpha
):

    xi = 0
    for s in A:
        xi += rewards[s] - Phi + low_bound_l + (N - (1 - alpha) * t) * high_bound_0

    return xi


def get_phi():
    return 0


def ocef(delta, alpha, epsilon, K):

    S = K
    # eps_no_envy = get_eps_no_envy()
    # envy = get_envy()
    N = np.zeros_like(S)
    rewards = []
    A = []

    t = 0
    while True:
        l = np.random.choice(S)

        betas, low_bounds, high_bounds = update_bounds(delta, N, rewards)

        Phi = get_phi()

        # not sure if t here is t-1 or the actual t in their formulas
        xi = get_conservarive_constraint(
            t, A, rewards, Phi, low_bounds[l], high_bounds[0], N[0], alpha
        )

        if betas[0] > min(betas[1:]) or xi < 0:
            k_t = 0
            N[0] += 1
        else:
            k_t = l
            N[l] += 1
            # TODO Store all t for which baseline was not pulled
            A.append(t)

        # Observe context, show action get reward
        # Update conf intervals
        # TODO CODE THIS

        # Store all rewards
        rewards.append[r]

        S = remove_non_envy_elements(S)

        t += 1
        if exists_higher_utility(S, k_t):
            return True  # envy
        if not S:
            return False  # eps_no_envy


