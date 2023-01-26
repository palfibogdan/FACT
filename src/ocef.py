import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# TODO:
# Implement conservative constraint Xi
# Implement observe, action and reward part
# Verify indices for Beta (Lemma 4 returns Beta_t but OCEF uses Beta_(t-1) )
# Implement get_phi from Lemma 5 and 6


def update_bounds(delta, N, rewards):

    sigma = 0.5
    omega = 0.99 

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
            beta = np.sqrt(
                (2 * sigma**2 * (1 + np.sqrt(omega)) ** 2 * (1 + omega))
            ) * np.sqrt(np.log(2 * (K + 1) / theta * np.log((1 + omega)))) + 0.01
        else:
            mean_mu = r / n
            beta = np.sqrt(
                (2 * sigma**2 * (1 + np.sqrt(omega)) ** 2 * (1 + omega)) / n
            ) * np.sqrt(np.log(2 * (K + 1) / theta * np.log((1 + omega) * n)))

        betas.append(beta)

        low_bounds.append(mean_mu - beta)
        high_bounds.append(mean_mu + beta)

    return betas, low_bounds, high_bounds


def remove_non_envy_elements(S, low_bounds, high_bounds, epsilon):
    new_S = []
    for k in S:
        if high_bounds[k] > low_bounds[0] + epsilon:
            new_S.append(k)

    return new_S


def exists_higher_utility(S, low_bounds, high_bounds):
    
    for k in S:
        if low_bounds[k] > high_bounds[0]:
            return True
    return False


def get_conservarive_constraint(
    t, A, reward_history, Phi, low_bound_l, high_bound_0, N, alpha
):

    xi = 0
    for s in A:
        xi += reward_history[s] - Phi + low_bound_l + (N - (1 - alpha) * t) * high_bound_0

    return xi


def get_phi(N, delta, betas):
    A = np.sum(N[1:])
    
    if A == 0:
        small_phi = 0
    else:
        small_phi = 1/2 * np.sqrt(2 * A * np.log(6 * A**2/delta) + 2/3 * np.log(6 * A**2/delta))
    
    sum_K = np.sum(np.multiply(N, betas))
    
    return min(sum_K, small_phi)


def ocef(delta, alpha, epsilon, S, utilities):

    N = np.zeros(len(S) + 1)
    rewards = np.zeros(len(S) + 1)
    reward_history = []
    A = []

    t = 0
    while True:
        l = np.random.choice(S)

        betas, low_bounds, high_bounds = update_bounds(delta, N, rewards)
        
        Phi = get_phi(N, delta, betas)

        # history of rewards???
        # not sure if t here is t-1 or the actual t in their formulas
        xi = get_conservarive_constraint(t, A, reward_history, Phi, low_bounds[l], high_bounds[0], N[0], alpha)

        if betas[0] > min(betas[1:]) or xi < 0:
            k_t = 0
            N[0] += 1
        else:
            k_t = l
            N[l] += 1
            #Store all t for which baseline was not pulled
            A.append(t)

        # Observe context, show action get reward

        r = utilities[k_t]
        # Store all rewards
        rewards[k_t] += r
        reward_history.append(r)
        # print("t: ", t)	
        # print("betas: ", betas)
        # print("low_bounds: ", len(low_bounds))
        # print("high_bounds: ", len(high_bounds))
        # print("N: ", N)
        # print("xi: ", xi)
        # print("S: ", S)
        # print("\n")
        S = remove_non_envy_elements(S, low_bounds, high_bounds, epsilon)
        
        #print("t: ", t)	
        # print("betas: ", betas)
        # print("low_bounds: ", low_bounds)
        # print("high_bounds: ", high_bounds)
        #print("N: ", N)
        #print("xi: ", xi)
        print("S: ", S)
        # print("\n")

        t += 1
        # if t%1000 == 0:
        #     print(t)
        if exists_higher_utility(S, low_bounds, high_bounds):
            return True, t, reward_history  # envy
        if not S:
            return False, t, reward_history  # eps_no_envy


def plot_duration(alphas, duration_per_problem):
    for problem, duration  in enumerate(duration_per_problem):
        plt.plot(alphas, problem, label=problem)
    plt.legend()
    plt.x_label("alpha")
    plt.y_label("duration")
    plt.show()

def main():
    S = [i for i in range(1, 10)]
    alphas = [i / 10 for i in range(1, 6)]
    problems = []

    # Problem 1
    problems.append([0.6] + [0.3] * 9)
    
    # Problem 2
    problems.append([0.3] + [0.6] + [0.3] * 8)
    
    # Problem 3
    problems.append([0.7 - 0.7 * (k / 10) ** 0.6 for k in range(10)])

    # Problem 4
    utilities = [0.7 - 0.7 * (k / 10) ** 0.6 for k in range(10)]
    utilities[0], utilities[1] = utilities[1], utilities[0]
    problems.append(utilities)
    
    durations = []
    costs = []
    
    _, t, _ = ocef(delta=0.05, alpha=0.1, epsilon=0.05, S=S, utilities=problems[1])
    print(t)

    # for utilities in tqdm(problems):
    #     temp_t = []
    #     temp_cost = []
    #     for alpha in tqdm(alphas):
    #         _, t, rewards = ocef(delta=0.05, alpha=alpha, epsilon=0.05, S=S, utilities=utilities)
    #         temp_t.append(t)
    #         # print(sum(rewards))
    #         temp_cost.append(t * utilities[0] - sum(rewards))
    #     durations.append(temp_t)
    #     costs.append(temp_cost)
    
    # #save durations
    # with open("durations.txt", "w") as f:
    #     for duration in durations:
    #         f.write(str(duration) + "\n")

    # #save costs
    # with open("costs.txt", "w") as f:
    #     for cost in costs:
    #         f.write(str(cost) + "\n")


    #plot_duration(alphas, durations)


if __name__ == "__main__":
    main()