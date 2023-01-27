import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# TODO:
# Implement conservative constraint Xi
# Implement observe, action and reward part
# Verify indices for Beta (Lemma 4 returns Beta_t but OCEF uses Beta_(t-1) )
# Implement get_phi from Lemma 5 and 6

# check out robin initialization
# problem probably lies in the values of the bounds of the arms not pulled
def update_bounds(delta, N, rewards):

    sigma = 0.5
    #omega = 1
    omega = 0.99

    theta = np.log(1 + omega) * ((omega * delta) / (2 * (2 + omega))) ** (
        1 / (1 + omega)
    )

    betas = []
    low_bounds = []
    high_bounds = []

    K = len(N)
    for n, r in zip(N, rewards):
        # if n == 0:
        #     mean_mu = 0
        #     beta = np.sqrt(
        #         (2 * sigma**2 * (1 + np.sqrt(omega)) ** 2 * (1 + omega))
        #     * np.log(2 * (K + 1) / theta * np.log((1 + omega)))) + 0.1
        # else:
        #     mean_mu = r / n
        #     beta = np.sqrt(
        #         (2 * sigma**2 * (1 + np.sqrt(omega)) ** 2 * (1 + omega)) / n
        #     * np.log(2 * (K + 1) / theta * np.log((1 + omega) * n)))


        mean_mu = r / n
        beta = np.sqrt(
            (2 * sigma**2 * (1 + np.sqrt(omega)) ** 2 * (1 + omega)) / n
        * np.log(2 * (K + 1) / theta * np.log((1 + omega) * n)))
            
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
        # changed this, not much difference
        small_phi = 1/2 * np.sqrt(2 * A * np.log(6 * A**2/delta)) + 2/3 * np.log(6 * A**2/delta)
    
    sum_K = np.sum(np.multiply(N, betas))
    
    return min(sum_K, small_phi)


def ocef(delta, alpha, epsilon, S, utilities):

    N = [0] * (len(S) + 1)
    #N = np.ones(len(S) + 1)
    rewards = np.zeros(len(S) + 1)
    #rewards = utilities
    reward_history = []
    A = []

    # sample each arm once
    for _ in range(2):
        N[0] += 1
        rewards[0] += utilities[0]
        reward_history.append(utilities[0])
        for k in S:
            N[k] += 1
            r = utilities[k]
            rewards[k] += r
            reward_history.append(r)

    t = len(reward_history)
    while True:
        l = np.random.choice(S)

        betas, low_bounds, high_bounds = update_bounds(delta, N, rewards)
        
        Phi = get_phi(N, delta, betas)

        # history of rewards???
        # not sure if t here is t-1 or the actual t in their formulas
        xi = get_conservarive_constraint(t, A, reward_history, Phi, low_bounds[l], high_bounds[0], N[0], alpha)

      
        if xi < 0 or betas[0] > min(betas[1:]):
            k_t = 0
        else:
            k_t = l
            #Store all t for which baseline was not pulled
            A.append(t)

        N[k_t] += 1
        # Observe context, show action get reward

        r = utilities[k_t]
        # Store all rewards
        rewards[k_t] += r
        # optimization problem
        reward_history.append(r)
       
        S = remove_non_envy_elements(S, low_bounds, high_bounds, epsilon)
        
        # betas = [round(beta, 2) for beta in betas]
        # print("betas: ", betas)
        # lows = [round(low, 2) for low in low_bounds]
        # highs = [round(high, 2) for high in high_bounds]
        # print("low_bounds: ", lows)
        # print("high_bounds: ", highs)
        # print("\n")
        #print("N: ", N)
        #print("xi: ", xi)
        #print(S)
        # print("\n")

        t += 1
        # if t%1000 == 0:
        #     print(t)
        if exists_higher_utility(S, low_bounds, high_bounds):
            return True, t, reward_history # envy
        if not S:
            return False, t, reward_history # eps_no_envy

        # early stopping
        if t >= 50000:
            return False, t, reward_history


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
    
    # _, t, _, = ocef(delta=0.05, alpha=0.1, epsilon=0.05, S=S, utilities=problems[3])
    # print(t)
    
    # starting betas
    # betas:  [4.67, 4.67, 4.67, 4.67, 4.67, 4.67, 4.67, 4.67, 4.67, 4.67]

    # good betas:
    # betas:  [0.07, 0.27, 0.26, 0.28, 0.26, 0.28, 0.27, 0.26, 0.26, 0.25]

    # TO Try:
    # 1. Change the starting betas DONE
    
    # Some arms are never pulled in problem 2
    # Arms that were pulled have a higher chance 


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
    
    # print("Problem 1")
    # for alpha in tqdm(alphas):
    #     temp_t = []
    #     temp_cost = []
    #     for _ in range(100):
    #         _, t, rewards = ocef(delta=0.05, alpha=alpha, epsilon=0.05, S=S, utilities=problems[0])
    #         temp_t.append(t)
    #         temp_cost.append(t * problems[0][0] - sum(rewards))

    #     mean_t = sum(temp_t) / len(temp_t)
    #     mean_cost = sum(temp_cost) / len(temp_cost)
    #     tp = (mean_t, mean_cost)
    #     # save to file
    #     with open(f"results/problem1.txt", "a") as f:
    #         f.write(str(tp) + "\n")

    # print("Problem 3")
    # for alpha in tqdm(alphas):
    #     temp_t = []
    #     temp_cost = []
    #     for _ in range(100):
    #         _, t, rewards = ocef(delta=0.05, alpha=alpha, epsilon=0.05, S=S, utilities=problems[2])
    #         temp_t.append(t)
    #         temp_cost.append(t * problems[2][0] - sum(rewards))

    #     mean_t = sum(temp_t) / len(temp_t)
    #     mean_cost = sum(temp_cost) / len(temp_cost)
    #     tp = (mean_t, mean_cost)
    #     # save to file
    #     with open(f"results/problem3.txt", "a") as f:
    #         f.write(str(tp) + "\n")
    
    # print("Problem 4")
    # for alpha in tqdm(alphas):
    #     temp_t = []
    #     temp_cost = []
    #     for _ in range(100):
    #         _, t, rewards = ocef(delta=0.05, alpha=alpha, epsilon=0.05, S=S, utilities=problems[3])
    #         temp_t.append(t)
    #         temp_cost.append(t * problems[3][0] - sum(rewards))

    #     mean_t = sum(temp_t) / len(temp_t)
    #     mean_cost = sum(temp_cost) / len(temp_cost)
    #     tp = (mean_t, mean_cost)
    #     # save to file
    #     with open(f"results/problem4.txt", "a") as f:
    #         f.write(str(tp) + "\n")


    #plot_duration(alphas, durations)


if __name__ == "__main__":
    main()