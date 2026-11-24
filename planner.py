import numpy as np
from src.robotics_toolbox.core import SE3, SO3
from abc import ABC, abstractmethod

class Trajectory(ABC):

    @abstractmethod
    def length(self) -> float:
        pass

    @abstractmethod
    def get_step(self, pose : SE3) -> float:
        pass

    @abstractmethod
    def _get_derivative(self, t : float) -> np.ndarray:
        pass

    @abstractmethod
    def _get_position(self, t):
        pass

    @abstractmethod
    def inverse(self, flange_pose : SE3) -> float:
        pass

    def get_speed(self, t : float) -> float:
        return np.linalg.norm(self._get_derivative(t))

    def get_tangent(self, t : float) -> np.ndarray:
        v = self._get_derivative(t) # tangent
        return v/np.linalg.norm(v) 

    def get_pose(self, t: float) -> SE3:
        p = self._get_position(t)
        R = self.get_rotation(t)

        return SE3(p, R)

    def get_rotation(self, t : float) -> SO3:
        z = self.get_tangent(t)

        up = np.array([0, 0, 1.0])
        if abs(z @ up) > 0.99: # parallel
            up = np.array([0, 1.0, 0])

        x = np.cross(up, z)
        x = x / np.linalg.norm(x)

        y = np.cross(z, x)

        return SO3(np.column_stack((x,y,z)))

class Arc(Trajectory):
    """
    f(t) = [

    ]
    """
    def __init__(self, center : np.ndarray, radius : float, angle : float):
        pass

class Line(Trajectory):
    """
    A trajectory line segment primitive
    """
    def __init__(self, start_pt : SE3, end_pt : SE3):
        self.start_pt = start_pt.translation
        self.end_pt = end_pt.translation

        self.direction = self.end_pt - self.start_pt

    @classmethod
    def from_t(cls, start_t : np.ndarray, end_t : np.ndarray):
        return cls(SE3(start_t), SE3(end_t))

    def _get_derivative(self) -> np.ndarray:
        return self.direction

    def inverse(self, flange_pose : SE3) -> float:
        p = flange_pose.translation
        v = self.direction

        return float(((p - self.start_pt) @ v) / (v @ v))

    def length(self):
        return np.linalg.norm(self.end_pt - self.start_pt)

    def get_pose(self, t : float):
        p = self.start_pt + t * self.direction
        R = self.get_rotation(t)

        return SE3(p,  R)

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
        return SE3()

    def get_curvature(self) -> float:
        """
        Returns a curvature of the trajectory
        """
        return 0.0
    # === Private ===

    def _calc_point(self) -> SE3:
        return SE3()

    def _calc_derivative(self) -> SE3:
        return SE3()