### Training & validation data for recommender

Concerning the experiments from section 5.1 of the paper.
What data is the (simulated) recommender system trained on?

In Appendix C.2, it is mentioned that

> We then simulate a recommender system’s estimation of preferences 
> using low-rank matrix completion (Bell and Sejnowski 1995) on a 
> training sample of 70% of the whole "ground truth" preferences, 
> with hyperparameter selection on a 10% validation sample.

However, in Section 5.1, we read

>  [...] the simulated recommender system estimates relevance 
>  scores using low-rank matrix completion (Bell and Sejnowski 
>  1995) on a training sample of 20% of the ground truth
>  preferences, where the rated / played items are sampled
>  uniformly at random. 

We interpreted this as: divide the ground truth data for either 
dataset (Last.fm\/MovieLens) into 70/10/20% train/validation/test 
sets; uniformly sample 20% of the (training) ground truth 
preferences, setting the remaining 80% to 0 to obtain a sparse 
matrix of ground truth preferences; train a matrix factorization 
model on these masked ground truth preferences, with hyperparameters 
selection on the 10% ground truth validation set.

**Questions**:
1. Is the interpretation above exact, or what is it meant by 
   "training sample of 70% of the whole ground truth preferences" 
   in Appendix C.2? 
   If this interpretation is correct, what do we 
   do with the remaining 20% masked ground truth test set?
   Are the 20% preferences sampled uniformly *per user* or *on the 
   whole ground truth matrix*?
3. "hyperparameter selection on a 10% validation sample": what 
   evaluation metrics is used in the paper on the validation 
   sample? We are currently using either of Mean Average Precision, 
   Area Under Curve, Precision or Normalized Discounted Cumulative 
   Gain, as provided by `implicit/evaluation.pyx`.
4. To compute the (synthetic) ground truth preferences, we perform 
   matrix factorization using the Alternating Least Squares method 
   from the Implicit library, followed by matrix completion using 
   the computed user and item factors.
   Is there any fundamental difference between the methodology used 
   to obtain the ground truth preferences and the simulated 
   recommender system's preferences? The explicit reference to the 
   work of Bell and Sejnowski in the second case is the source of 
   our question; is this mention just referring to the theoretical 
   underpinning of low-rank matrix completion, or is does it imply 
   the usage of a different model to compute the recommender's 
   preferences?
5. Could it be possible to provide us the fitted hyperparameters 
   for the ground truth and recommender preferences for both 
   datasets for full reproducibility?

### MovieLens-1M dataset

When transforming the MovieLens dataset into an implicit feedback 
one, it is mentioned that

> Since setting ratings < 3 are usually considered as negative 
> (Wang et al. 2018), we set ratings < 3 to zero, resulting in a 
> dataset with preference values among {0, 3, 3.5, 4, 4.5, 5}.

Ratings from MovieLens are integer-valued. If we set all ratings 
< 3 to 0, we are left with rating values {0, 3, 4, 5}, which is 
impossible to remap to the set {0, 3, 3.5, 4, 4.5, 5} with greater 
cardinality.

However, this is possible if we remap ratings {1, 2, 3, 4, 5} to 
{3, 3.5, 4, 4.5, 5}, and keep the unknown ratings at 0. 

As a result, we obtain implicit preferences (only positive 
preferences) but we shifted the preferences range to positive 
ratings only. This means that that originally low preferences 
(negatives) still map to low preferences in the new regime, but we 
now have an implicit feedback dataset.

Is this what is meant by "setting ratings < 3 to 0"?

### Rewards generation process

Appendix C.2 reads:

> We generate binary rewards using a Bernoulli distribution with 
> expectation given by our ground truth. We consider no context in 
> these experiments, so that the policies and rewards only depend 
> on the user and the item.

This is how we understand the process so far:
1. Turn the ground truth preferences into probabilities using a 
   Softmax function with inverse temperature set to 5.
2. Consider each user-item preference probability as the 
   expectation of a Bernoulli distribution, effectively sampling 
   true recommendations from \#number of users $\times$ \#number of items 
   Bernoulli distributions.

This interpretation has 2 implications:
1. There could be **more than one true recommendation** for a user; 
   in this case, we give a reward of 1 if the user recommendation 
   overlaps with (at least one of) the true recommendations, and 0 
   otherwise.
2. There could be **no true recommendations** for a user. This happens 
   mostly when user preferences are smooth and all have low 
   values \[which could could occur in a substantial amount of 
   cases\, about 1/3 of the times upon manual inspection on the 
   Last.fm dataset]. In this case, we say that the user is 
   "neutral", and the reward is 0 regardless of the user 
   recommendation.

Is this interpretation correct?