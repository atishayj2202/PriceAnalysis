from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name, reliability_score, default_val):
        self.name = name
        self.reliability_score = reliability_score
        self.default_val = default_val

    @abstractmethod
    def assess(self, data_dict):
        """
        Assesses the specific factor based on the inputs provided.
        Returns a dictionary with keys:
          - 'factor_value': computed multiplier or modifier
          - 'r2': explanatory power
          - 'reliability': reliability string
          - 'status': usage status
          - 'details': explanatory details
        """
        pass
