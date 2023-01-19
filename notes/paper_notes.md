# Notes on ["Online Certification of preference-based fairness for personalize recommender systems"](https://arxiv.org/pdf/2104.14527.pdf) 

## Experiments
---
* Last.fm-2k dataset: https://grouplens.org/datasets/hetrec-2011/
* MovieLens-1m dataset: https://grouplens.org/datasets/movielens/1m/

### Sources of envy (sec 5.1)

#### Envy from model mispecifications (Appendix C.2)
---

* *Goal*: show that, if the modelling assumptions are too strong, the
  recommendations of a model are envious; intuitively, this occurs
  because the estimated user preferences are misaligned with the 
  true, unknown ones.
* *Background*: to make accurate recommendations, a system 
  should know the preference of each user for each item, given by 
  a complete user-item matrix $\mathbf{X}\in\mathbb{R}^{N\times M}$, with $N$ the number of users 
  and $M$ the number of items. 
  
  However, in reality such matrices are always incomplete: users 
  do not interact with every item, and as a result $\mathbf{X}$ is sparse. 
  
  To estimate the missing preferences in $\mathbf{X}$, [matrix factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) 
  is commonly used: the full, unobserved matrix $\mathbf{X}\approx \mathbf{X}^{\ast}=\mathbf{UV}^T$ is 
  approximated by the product of two matrices $\mathbf{U}\in\mathbb{R}^{N\times K}$ and $\mathbf{V}\in\mathbb{R}^{M\times K}$ 
  where $K$ is the the number of *factors* - the dimensionality of the 
  projected space, with $K>M$ (lower rank), effectively meaning 
  compression with resemblances to PCA. The factors can be computed 
  in different ways, such as SVD and SGD. 
  
  [Some useful slides on matrix factorization](https://cse.iitk.ac.in/users/piyush/courses/ml_autumn16/771A_lec14_slides.pdf) and an article on 
  [methods to compute factors](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf).
  
* In the paper, the authors use the implementation of 
  [Implicit Least Squares](https://implicit.readthedocs.io/en/latest/als.html?highlight=collaborative) from the Python Implicit library to 
  compute "dummy" (synthetic) *ground truth preferences* for both the 
  Last.fm and MovieLens datasets, which are then used by the 
  Bernoulli reward model to simulate real user feedback (a proxy 
  for utility) in the OCEF algorithm.
  
  *NOTE* A similar work is done the create the recommender system $\hat{\mathbf{X}}$, 
  but they mention "low-rank matrix completion" instead of matrix 
  factorization here; as I understand it, matrix completion is just 
  the multiplication $\mathbf{UV}^T$ that fills in the sparse matrix $\mathbf{X}$. 
  So what is really the difference between the ground truth user 
  preference matrix $\mathbf{X}^{\ast}$ and the recommender system's estimated 
  preferences $\hat{\mathbf{X}}$? Is $\hat{\mathbf{X}}$ meant to be an approximation of $\mathbf{X}^{\ast}$?

#### Envy from equal user utility EEU (Appendix C.3)
---