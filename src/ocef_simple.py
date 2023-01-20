import numpy as np

# Regarding the upper and lower boundaries of the utilities mu: 
# 1. Do you use Lemma 4 to calculate the boundaries or do you use some other method? If so, what is the value of omega?
# 2. We understand that the boundaries are recalculated at each time step t, but how is that done when the utilities are already given? (Appendix D.1)

def exploration_constraint(alpha, S, actions):
    # S = {"0":0.2, "1":0.3, "2":0.5} # utilities
    # actions = ["1", "0", "2"] # actions
    s = 0

    for a in actions:
        s += S[a]

    if len(actions) > 0:
        s /= len(actions)
    else:
        s = 0

    # continue (select arm k)
    if s >= (1-alpha) * S[0]:
        return True
    
    # stop (select baseline)
    return False


def update_arm_list(S, epsilon):

    to_remove = []
    for s in S.keys():
        if S[s] <= S[0] + epsilon and s != 0:
            to_remove.append(s)
    
    for s in to_remove:
        del S[s]

    return S

def ocef_simple(S, alpha=0.05, epsilon=0.05):
    
  
    S_copy = S.copy()

    t = 1
    chosen_arms = []

    while True:
        l = np.random.choice(list(S.keys()))

        if exploration_constraint(alpha, S_copy, chosen_arms):
            k_t = l
        else:
            k_t = 0

        chosen_arms.append(k_t)
        
        # Observe context , get reward, update conf intervals
        
        S = update_arm_list(S, epsilon)

        for s in S.keys():
            if S[s] > S[0]:
                return True, t # envy
        
        if len(S.keys()) == 1:
            return False, t # eps_no_envy
        
        t += 1

# problem 1
S = {0:0.6, 1:0.3, 2:0.3, 3:0.3, 4: 0.3, 5: 0.3, 6: 0.3, 7: 0.3, 8: 0.3, 9: 0.3}
print(ocef_simple(S))