from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from src.robotics_toolbox.core import SE3, SO3
from abc import ABC, abstractmethod

class Trajectory(ABC):

    @abstractmethod
    def length(self) -> float:
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
        return SE3(
            self._get_position(t),
            self._get_rotation(t)
        )

    def _get_rotation(self, t : float) -> SO3:
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
    def __init__(
            self, 
            start_pt : SE3,
            center : np.ndarray,
            axis : np.ndarray,
            angle : float
            ):
        self.start_pt = start_pt.translation
        self.center = center
        self.axis = axis / np.linalg.norm(axis)
        self.angle = angle

        self.radial = self.start_pt - self.center
        self.radius = np.linalg.norm(self.radial)

    def _get_position(self, t):
        R = SO3.exp(self.axis * self.angle * t)

        return self.center + R.act(self.radial)

    def _get_derivative(self, t: float) -> np.ndarray:
        radial = self._get_position(t) - self.center

        return self.angle * np.cross(self.axis, radial)

    def length(self) -> float:
        return float(self.radius * abs(self.angle))

    def inverse(self, flange_pose: SE3) -> float:
        u = flange_pose.translation - self.center

        alpha = np.arctan2(
            self.axis @ np.cross(self.radial, u),
            self.radial @ u
        )

        return float(alpha / self.angle)

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

    def _get_position(self, t):
        return self.start_pt + t * self.direction

    def _get_derivative(self, t) -> np.ndarray:
        return self.direction

    def inverse(self, flange_pose : SE3) -> float:
        p = flange_pose.translation
        v = self.direction

        return float(((p - self.start_pt) @ v) / (v @ v))

    def length(self):
        return np.linalg.norm(self.end_pt - self.start_pt)
    
tr1 :list[Trajectory] = [
    Line.from_t(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.2])
        )
]

tr2 : list[Trajectory] = [
    Line.from_t(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.08])
        ),
    
    Line.from_t(
        np.array([0.0, 0.0, 0.08]),
        np.array([0.07, 0.0, 0.15])
        ),

    Line.from_t(
        np.array([0.07, 0.0, 0.15]),
        np.array([0.07, 0.0, 0.2])
        )
]

tr3 : list[Trajectory] = [
    Line.from_t(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.05])
        ),
    
    Line.from_t(
        np.array([0.0, 0.0, 0.05]),
        np.array([-0.05, 0.0, 0.1])
        ),

    Line.from_t(
        np.array([-0.05, 0.0, 0.1]),
        np.array([-0.05, 0.05, 0.15])
        ),

    Line.from_t(
        np.array([-0.05, 0.05, 0.15]),
        np.array([-0.05, 0.05, 0.2])
        )
    
]

tr4 : list[Trajectory] = [
    Line.from_t(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.03])
        ),

    Line.from_t(
        np.array([0.0, 0.0, 0.03]),
        np.array([-0.015, 0.0, 0.045])
        ),

    Line.from_t(
        np.array([-0.015, 0.0, 0.045]),
        np.array([-0.015, 0.0, 0.105])
        ),

    Arc(
        start_pt=SE3(np.array([-0.015, 0.0, 0.105])),
        center=np.array([0.035, 0, 0.105]),
        axis=np.array([0, 1, 0]),
        angle=1.57
    ),

    Line.from_t(
        np.array([0.035, 0, 0.155]),
        np.array([0.085, 0, 0.155])
    )
]

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


def test_trajectory(tr: Trajectory):
    for t in np.linspace(0, 1, 11):
        pose = tr.get_pose(t)
        t_inv = tr.inverse(pose)

        assert np.isclose(t, t_inv)

    print("Trajectory OK")

def plot_trajectory(trajectory: list[Trajectory], n: int = 100):
    fig = plt.figure(figsize=(12, 10))

    ax3d = fig.add_subplot(221, projection='3d')
    ax_xy = fig.add_subplot(222)
    ax_xz = fig.add_subplot(223)
    ax_yz = fig.add_subplot(224)

    ax_xy.invert_yaxis()

    for i, segment in enumerate(trajectory):
        points = np.array([
            segment.get_pose(t).translation
            for t in np.linspace(0, 1, n)
        ])

        x, y, z = points.T

        label = f"{type(segment).__name__} {i}"

        ax3d.plot(x, y, z, label=label)
        ax_xy.plot(x, y)
        ax_xz.plot(x, z)
        ax_yz.plot(y, z)

    ax3d.set_title("3D trajectory")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_aspect("equal")
    ax3d.legend()

    ax3d.set_xlim(-0.1, 0.1)
    ax3d.set_ylim(-0.1, 0.1)
    ax3d.set_zlim(0.0, 0.2)

    ax3d.set_box_aspect([1, 1, 1])

    for ax, a, b in [
        (ax_xy, "X", "Y"),
        (ax_xz, "X", "Z"),
        (ax_yz, "Y", "Z")
    ]:
        ax.set_title(a + b)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_aspect("equal")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

plot_trajectory(tr4)