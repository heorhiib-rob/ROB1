import numpy as np
from src.robotics_toolbox.core import SE3

class Planner:
    """
    """

    def __init__(self):
        pass

    # === Public ====
    def update_progress(self, hoop_pose : SE3) -> None:
        """
        Updates inner variable 'progress' based on hoops pose
        """
        pass

    def get_next_pose(self) -> SE3:
        """
        Returns a pose of TCP as a next step
        """
        pass

    def get_curvature(self) -> float:
        """
        Returns a curvature of the trajectory
        """
        pass
    # === Private ===

    def _calc_point(self) -> SE3:
        pass

    def _calc_derivative(self) -> SE3:
        pass