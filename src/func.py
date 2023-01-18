import numpy as np

def calc_bounds(delta, omega):
    
    # where are omega, sigma, K specified?

    theta = np.log(1+omega) * ((omega*delta) / (2*(2+omega)))**(1/(1+omega))
    
    #N = count?

    #mean_mu =

    beta = np.sqrt((2 * sigma**2 * (1+np.sqrt(omega))**2 * (1+omega)) / N) * np.sqrt(np.log(2*(K+1)/theta * np.log((1+omega)*N))) 

    low_bounds = mean_mu - beta
    high_bounds = mean_mu + beta

    return beta, low_bounds, high_bounds

def get_min_beta():
	return 0

def remove_non_envy_elements(S):
    return S

def get_eps_no_envy():
	return 0

def get_envy():
	return 1

def exists_higher_utility(S, k_t):
	return True

def ocef(delta, alpha, epsilon, K):

	S = K
	eps_no_envy = get_eps_no_envy()
	envy = get_envy()

	beta0, low_bounds, high_bounds = calc_bounds()

	t = 0
	while(True):
		l = np.random.choice(S)

		if beta0[t] > get_min_beta() or xi[t] < 0:
			k_t = 0
		else:
			k_t = l


		# Observe context, show action get reward
		# Update conf intervals
		# TODO CODE THIS

		S = remove_non_envy_elements(S)

		if exists_higher_utility(S, k_t):
			return envy
		if not S:
			return eps_no_envy


