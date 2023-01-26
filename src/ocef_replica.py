import random

import numpy as np

# TODO:
# Implement observe, action and reward part
# Verify indices for Beta (Lemma 4 returns Beta_t but OCEF uses Beta_(t-1) )


# def update_bounds(delta, N, rewards):

#     # where is omega specified?
#     sigma = 0.5
#     omega = 0.5  # random in (0,1)

#     theta = np.log(1 + omega) * ((omega * delta) / (2 * (2 + omega))) ** (
#         1 / (1 + omega)
#     )

#     betas = []
#     low_bounds = []
#     high_bounds = []
#     K = len(N)
#     for n, r in zip(N, rewards):
#         if n == 0:
#             mean_mu = 0
#         else:
#             mean_mu = r / n

#         beta = np.sqrt(
#             (2 * sigma**2 * (1 + np.sqrt(omega)) ** 2 * (1 + omega)) / N
#         ) * np.sqrt(np.log(2 * (K + 1) / theta * np.log((1 + omega) * N)))
        
#         betas.append(beta)

#         low_bounds.append(mean_mu - beta)
#         high_bounds.append(mean_mu + beta)

#     return beta, low_bounds, high_bounds


def update_bounds(bounds, N, rewards):

    """
    bounds = [(low_bounds, high_bounds)]
    """

    betas = []
    for low, high in bounds:
        betas.append((low + high) / 2)

    for i in range(len(rewards)):
        if N[i] == 0:
            mean_mu = 0
        else:
            mean_mu = rewards[i] / N[i]
        
        low = mean_mu - betas[i]
        high = mean_mu + betas[i]
        bounds[i] = (low, high)
    
    return bounds, betas


def remove_non_envy_elements(S, low_bounds, high_bounds, epsilon):
    new_s = []
    for k in range(len(S)):
        if high_bounds[k] > low_bounds[0] + epsilon:
            new_s.append(S[k])
    return new_s


def exists_higher_utility(S, low_bounds, high_bounds):
    for k in range(len(S)):
        if low_bounds[k] > high_bounds[0]:
            return True
    return False


def get_conservarive_constraint(t, A, rewards, Phi, low_bound_l, high_bound_0, N, alpha):

    xi = 0
    for s in A:
        xi += rewards[s] - Phi + low_bound_l + (N - (1 - alpha) * t) * high_bound_0

    return xi


def get_phi(N, delta, betas):
    A = np.sum(N[1:])
    
    if A == 0:
        small_phi = 0
    else:
        small_phi = 1/2 * np.sqrt(2 * A * np.log(6 * A**2/delta) + 2/3 * np.log(6 * A**2/delta))
    
    sum_K = np.sum(np.multiply(N, betas))
    
    return min(sum_K, small_phi)

def ocef(delta, alpha, epsilon, S):

    N = np.zeros_like(S)
    rewards = np.zeros_like(S)
    A = []
    
    bounds = [(0, 1) for _ in range(len(S))]

    t = 0
    while True:
        l = np.random.choice(range(len(S)))
        
        bounds, betas = update_bounds(bounds, N, rewards)

        Phi = get_phi(N, delta, betas)

        # not sure if t here is t-1 or the actual t in their formulas
        xi = get_conservarive_constraint(t, A, rewards, Phi, bounds[l][0], bounds[0][1], N[0], alpha)

        if betas[0] > min(betas[1:]) or xi < 0:
            k_t = 0
            N[0] += 1
        else:
            k_t = l
            N[l] += 1
            A.append(t)

        # Observe context, show action get reward
        # Update conf intervals
        # TODO CODE THIS

        # Assuming reward = utility, pick reward of decision k_t
        r = S[k_t]

        # Store all rewards
        rewards[k_t] += r

        lowers = [low for low, _ in bounds]
        uppers = [high for _, high in bounds]
        S = remove_non_envy_elements(S, lowers, uppers, epsilon)

        t += 1

        if exists_higher_utility(S, lowers, uppers):
            return True, t  # envy
        if not S:
            return False, t  # eps_no_envy


def main():

    S = [0.6] + [0.3] * 9
    #S = {i: v for i, v in enumerate(values)}

    print(ocef(delta=0.05, alpha=0.4, epsilon=0.05, S=S))


if __name__ == "__main__":
    main()