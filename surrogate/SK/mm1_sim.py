__all__ = ["MM1sim"]

import numpy as np


def MM1sim(service_rate, arrival_rate, n, runlength, init):
    """
    Simulate customer waiting times in an M/M/1 queue.

    Parameters:
    service_rate : array_like
        The rate at which the service station can service customers. Can be a scalar or a vector.
    arrival_rate : float
        The rate at which customers arrive at the service station.
    n : array_like
        Number of replications at each design point.
    runlength : int
        Number of customers per simulation run.
    init : {'stationary', 'mean', 'zero'}
        Method for initializing the simulation:
        - 'stationary': Initialize in steady state.
        - 'mean': Initialize with the mean waiting time.
        - 'zero': Initialize with zero waiting time.

    Returns:
    Y : ndarray
        Vector of mean waiting times at each design point.
    Vhat : ndarray
        Vector of variances of the waiting times at each design point.

    Raises:
    ValueError
        If the queue is unstable (service rate < arrival rate).

    Notes:
    This function simulates an M/M/1 queueing system and calculates the mean and variance
    of the waiting time for a given service and arrival rate. The simulation can be
    initialized in different ways based on the 'init' parameter.
    """

    if np.min(service_rate) < arrival_rate:
        raise ValueError('Unstable queue.')

    k = len(service_rate)  # number of design points
    service_mean = 1 / service_rate  # Able to broadcast given proper dimension
    arrival_mean = 1 / arrival_rate  # Should not broadcast
    load = service_mean / arrival_mean
    truth = load / (service_rate - arrival_rate)   # Theoretical mean waiting time

    waits = [np.zeros(n_i) for n_i in n]   # Initialize waiting times for each design point

    # Initialize waiting times based on init method
    if init == 'stationary':
        emean = 1. / (service_rate - arrival_rate)  # Mean waiting time for exponential distribution
        for m in range(k):
            # Set initial waiting time based on stationary distribution
            waits[m] = np.where(np.random.rand(n[m]) < load[m],
                                np.random.exponential(emean[m], n[m]),
                                0)
    elif init == 'mean':
        # Set all initial waiting times to the theoretical mean
        for m in range(k):
            waits[m] = np.full(n[m], truth[m])
    elif init == 'zero':
        # Start with no waiting time
        for m in range(k):
            waits[m] = np.zeros(n[m])
    else:
        raise ValueError('Invalid initialization method.')

    # Simulate waiting times
    for m in range(k):
        # Generate random service and arrival times
        services = np.random.exponential(service_mean[m], (runlength, n[m]))
        arrivals = np.random.exponential(arrival_mean, (runlength, n[m]))

        # Calculate waiting time for each customer
        wait = waits[m]
        for i in range(1, runlength):
            wait = np.maximum(0, wait + services[i, :] - arrivals[i, :])
            waits[m] += wait

        # Average waiting time over the runlength
        waits[m] /= runlength

    # Compute the mean and variance of waiting times for each design point
    Y = np.array([np.mean(waits[m]) for m in range(k)])
    Vhat = np.array([np.var(waits[m], ddof=1) / n[m] for m in range(k)])

    return Y, Vhat
