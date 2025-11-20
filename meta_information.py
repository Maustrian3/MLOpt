import pandas as pd
from matplotlib import pyplot as plt

class MetaInformation:
    def __init__(self):
        self.iterations = 0
        self.obj_per_iteration = []
        self.final_obj = 0

    def inc_iterations(self):
        """Set the number of iterations."""
        self.iterations += 1

    def set_final_obj(self, obj):
        """Set the objective value."""
        self.final_obj = obj

    def add_iteration_data(self, obj_for_iteration):
        """
        Add data for an iteration: objective value.
        :param obj_for_iteration: Objective value for the iteration.
        """
        self.obj_per_iteration.append(obj_for_iteration)

    @staticmethod
    def average(meta_infos):
        """
        Compute the average of multiple MetaInformation objects.
        :param meta_infos: List of MetaInformation objects.
        :return: A new MetaInformation object with averaged values.
        """
        if not meta_infos:
            raise ValueError("The list of MetaInformation objects is empty.")

        avg_meta = MetaInformation()
        n = len(meta_infos)

        # Aggregate
        avg_meta.iterations = sum(m.iterations for m in meta_infos) // n  # Use integer division for iterations
        avg_meta.final_obj = sum(m.final_obj for m in meta_infos) / n

        # Find the minimum length
        min_obj_length = min(len(m.obj_per_iteration) for m in meta_infos)

        # Compute averages for obj_per_iteration up to the shortest list length
        avg_meta.obj_per_iteration = [
            sum(m.obj_per_iteration[i] for m in meta_infos) / n
            for i in range(min_obj_length)
        ]

        return avg_meta


class MetaInformationPlotter:
    @staticmethod
    def plot_meta_info(meta_infos: [MetaInformation], title_prefix=""):


        # Plot objective values per iteration
        plt.figure(figsize=(10, 5))

        for meta in meta_infos:
            plt.plot(
                meta.obj_per_iteration,
                linewidth=2.0,  # thinner line
                alpha=1.0,  # more transparent
                marker = 'o',
            )

        # avg_meta = MetaInformation.average(meta_infos)
        # plt.plot(
        #     avg_meta.obj_per_iteration,
        #     linewidth=2.0,  # thicker line
        #     marker='o',
        #     label='Average objective'
        # )

        plt.title(f"{title_prefix}Objective Per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()