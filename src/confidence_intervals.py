import numpy as np


def get_probs_of_p_hat(sample_size, num_successes, ps=np.linspace(1 / 100, 1, 100), n_trials=1000):
    """
    sample_size: int
    num_successes: int or np.array of int, 0 <= num_successes <= sample_size
    ps: proportion parameters to use as prior. defaults to uniform prior with 100 samples
    n_trials: number of trials
    """
    probs_of_p_hat = np.empty((len(ps), len(num_successes))) if isinstance(num_successes, np.ndarray) else np.empty(
        ps.shape)
    for i, p in enumerate(ps):
        successes = np.zeros(num_successes.shape) if isinstance(num_successes, np.ndarray) else 0
        for trial in range(n_trials):
            outcome = np.random.binomial(n=sample_size, p=p)
            successes += (outcome == num_successes)
        if isinstance(num_successes, np.ndarray):
            probs_of_p_hat[i, :] = successes / n_trials
        else:
            probs_of_p_hat[i] = successes / n_trials

    return probs_of_p_hat


def get_CI(x, probs, confidence=0.95):
    """returns the confidence interval of x, where probs[i] == P(x[i] <= x < x[i+1])
    pre: probs sums to 1 and length is one less than len(x)"""
    tail_area = (1 - confidence) / 2
    assert 1 - tail_area / 100 < sum(probs) < 1 + tail_area / 100, str(sum(probs))

    lower = 0
    prev_total = 0
    total = probs[lower]
    while total < tail_area:
        prev_total = total
        lower += 1
        total += probs[lower]

    # linearly interpolate the "exact" x
    lowerx = x[lower] + (x[lower + 1] - x[lower]) * (tail_area - prev_total) / (total - prev_total)
    upper = lower

    while total < 1 - tail_area:
        prev_total = total
        upper += 1
        total += probs[upper]

    upperx = x[upper] + (x[upper + 1] - x[upper]) * (1 - tail_area - prev_total) / (total - prev_total)
    return lowerx, upperx


def get_CI_of_p(sample_size, num_successes, p_edges=np.linspace(0, 1, 101), n_trials=10_000, confidence=0.95):
    """get the `confidence`*100% confidence interval for the population proportion given that num_successes of
    sample_size individuals were successes."""
    # if sample_size == 0:
    #     pass
    #     return (np.nan, np.nan)
    ps = (p_edges[1:] + p_edges[:-1]) / 2
    probs_of_p_hat = get_probs_of_p_hat(sample_size, num_successes, ps=ps, n_trials=n_trials)
    if sum(probs_of_p_hat) == 0:
        print(probs_of_p_hat, sample_size, num_successes)
    probs_of_p = probs_of_p_hat / np.sum(probs_of_p_hat, axis=0)
    return get_CI(p_edges, probs_of_p, confidence=confidence)
