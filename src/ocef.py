import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# TODO:
# Implement conservative constraint Xi
# Implement observe, action and reward part
# Verify indices for Beta (Lemma 4 returns Beta_t but OCEF uses Beta_(t-1) )
# Implement get_phi from Lemma 5 and 6

sigma = 0.4

# check out robin initialization
# problem probably lies in the values of the bounds of the arms not pulled
def update_bounds(delta, N, rewards):
    
    omega = 0.01

    # original
    # sigma = 0.5 (deduced from paper)
    # omega = 0.99 (confirmed by authors)

    theta = np.log(1 + omega) * ((omega * delta) / (2 * (2 + omega))) ** (
        1 / (1 + omega))

    betas = []
    low_bounds = []
    high_bounds = []

    K = len(N) - 1
    for n, r in zip(N, rewards):
        
        if n == 0:
            # calculate bounds as if they were chosen once
            mean_mu = 0
            n = 1
        else:
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
        xi += reward_history[s]

    xi = xi - Phi + low_bound_l + (N - (1 - alpha) * t) * high_bound_0

    return xi


def get_phi(N, delta, betas):
    A = np.sum(N[1:])
    
    if A == 0:
        small_phi = 0
    else:
        small_phi = sigma * np.sqrt(2 * A * np.log(6 * A**2/delta)) + 2/3 * np.log(6 * A**2/delta)
    
    sum_K = np.sum(np.multiply(N[1:], betas[1:]))
    
    return min(sum_K, small_phi)


def ocef(delta, alpha, epsilon, S, means):
    
    N = [0] * (len(means))
    rewards = [0] * (len(means))
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

      
        if xi < 0 or betas[0] > min(betas[1:]):
            k_t = 0
        else:
            k_t = l
            #Store all t for which baseline was not pulled
            A.append(t)

        N[k_t] += 1

        # rewards are given by a bernoulli distribution
        r = np.random.binomial(1, means[k_t])
        
        # Store all rewards
        rewards[k_t] += r

        # optimization problem
        reward_history.append(r)

        S = remove_non_envy_elements(S, low_bounds, high_bounds, epsilon)
        
        #prints(betas, low_bounds, high_bounds, N, xi, S, rewards, t)

        t += 1
        
        if exists_higher_utility(S, low_bounds, high_bounds):
            return True, t, reward_history # envy
        if not S:
            return False, t, reward_history # eps_no_envy

        # early stopping
        if t >= 70000:
            return False, t, reward_history


def prints(betas, low_bounds, high_bounds, N, xi, S, rewards, t):

    #clear screen
    #os.system('cls' if os.name == 'nt' else 'clear')
    
    # betas = [round(beta, 2) for beta in betas]
    # print("betas: ", betas)
    # lows = [round(low, 2) for low in low_bounds]
    # highs = [round(high, 2) for high in high_bounds]
    # print("low_bounds: ", lows)
    # print("high_bounds: ", highs)
    #print("\n")
    # print("N: ", N)
    # print("xi: ", xi)
    # print(S)
    # print("\n")
    # print(rewards)
    if t%1000 == 0:
            print(t)
    

def plot(alphas):

    # save as npy next time...

    all_durations = []
    all_costs = []

    for i in range(1, 5):
        # read from file
        with open(f"src/results_ocef/problem{i}.txt", "r") as f:
            data = f.read()
        
        data = data.split("\n")
        #remove brackets
        data = [d[1:-1] for d in data]
        #remove empty strings
        data = [d for d in data if d]
        #split into tuples
        data = [d.split(", ") for d in data]
        #convert to floats
        data = [(float(d[0]), float(d[1])) for d in data]
        
        durations = [d[0] for d in data]
        costs = [d[1] for d in data]
        all_durations.append(durations)
        all_costs.append(costs)

    
    # plot durations and costs side by side\
    fig, (ax1, ax2) = plt.subplots(1, 2)
    colors = ["red", "green", "blue", "cyan"]
    styles = ["-", "--", "-.", ":"]
    for durations, costs in zip(all_durations, all_costs):
        c = colors.pop(0)
        style = styles.pop(0)
        durations = np.array(durations)
        ax1.plot(alphas, durations, color=c, linestyle=style)
        ax2.plot(alphas, costs, color = c, linestyle=style)
    
    ax1.set_xlabel("alpha")
    ax2.set_xlabel("alpha")
    ax1.set_ylabel("duration")
    ax2.set_ylabel("average cost of exploration")
    
    ax1.legend(["1", "2", "3", "4"])
    ax2.legend(["1", "2", "3", "4"])

    # change figure size
    fig.set_size_inches(10, 5)
    # make plots square
    fig.tight_layout()
    # set y axis to 10^4 for durations
    ax1.set_yscale("log")

    # add all y ticks for ax1
    ax1.set_yticks([1000.0, 5000.0, 10000.0, 50000.0])
    ax1.set_yticklabels([0.1, 0.5, 1.0, 5.0])
    
    # write "x10^4" on left top of ax1
    ax1.text(0.01, 1.03, "x10^4", transform=ax1.transAxes, fontsize=10, verticalalignment='top')
    
    ax1.grid()
    ax2.grid()
    # ax1.set_yticks([0.1, 0.5, 1.0, 5.0])
    # ax1.set_yticklabels([0.1, 0.5, 1.0, 5.0])
   
    plt.show()


def run_experiment(S, alphas, problems, num_runs=100):

    for problem, means in enumerate(problems):
        print("Problem ", problem)
        for alpha in tqdm(alphas):
            temp_t = []
            temp_cost = []
            for _ in range(num_runs):
                _, t, rewards = ocef(delta=0.05, alpha=alpha, epsilon=0.05, S=S, means=means)
                temp_t.append(t)
                temp_cost.append(t * means[0] - sum(rewards))
            print("Timed out:", len([t for t in temp_t if t >= 70000]))
            mean_t = sum(temp_t) / len(temp_t)
            mean_cost = sum(temp_cost) / len(temp_cost)
            tp = (mean_t, mean_cost)
            # save to file
            with open(f"src/results_ocef/problem{problem}.txt", "a") as f:
                f.write(str(tp) + "\n")


def main():
    S = [i for i in range(1, 10)]
    #alphas = [i / 10 for i in range(1, 6)]
    alphas = np.linspace(0.01, 0.5, 12)
    problems = []
    
    # Problem 1
    problems.append([0.6] + [0.3] * 9)
    
    # Problem 2
    problems.append([0.3] + [0.6] + [0.3] * 8)
    
    # Problem 3
    problems.append([0.7 - 0.7 * (k / 10) ** 0.6 for k in range(10)])

    # Problem 4
    means = [0.7 - 0.7 * (k / 10) ** 0.6 for k in range(10)]
    means[0], means[1] = means[1], means[0]
    problems.append(means)
    
    prob = 3
    # temp_t = []
    # temp_cost = []
    # for _ in tqdm(range(100)):
    #     _, t, rewards, = ocef(delta=0.05, alpha=0.4, epsilon=0.05, S=S, means=problems[prob])
    #     temp_t.append(t)
    #     temp_cost.append(t * problems[prob][0] - sum(rewards))
    
    # print("Avg duration:", sum(temp_t) / len(temp_t))
    # print("Avg cost:", sum(temp_cost) / len(temp_cost))
    # print("Timed out:", len([t for t in temp_t if t >= 70000]))
    # print("Problem ", prob+1)
    # print(problems[prob])
    


    #run_experiment(S, alphas, problems, num_runs=100)

    plot(alphas)


if __name__ == "__main__":
    main()