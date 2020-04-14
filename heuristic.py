import numpy as np


class NetworkHeuristic(object):
    def __init__(self, threash_hold=13):
        self.thresh_hold = threash_hold
        self.global_aware_subpolicy_index = 0
        self.local_only_subpolicy_index = 1

    def predict(self, observation):
        network_state = observation[-1]
        if network_state * 10 < self.thresh_hold:
            return self.global_aware_subpolicy_index, None
        else:
            return self.local_only_subpolicy_index, None

