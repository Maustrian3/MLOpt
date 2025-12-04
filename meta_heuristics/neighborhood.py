from abc import ABC, abstractmethod

from meta_heuristics.problem_instance import ProblemInstance


class Neighborhood(ABC):
    """Defines the interface for neighborhood structures that generate and apply moves."""

    def __init__(self):
        pass

    @abstractmethod
    def generate_random_neighborhood_move(self, problem_inst: ProblemInstance, schedule, conflicts):
        pass

    @abstractmethod
    def apply_neighborhood_move(self, schedule, conflicts):
        pass