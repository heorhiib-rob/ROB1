import cv2
import numpy as np
from src.robotics_toolbox.core.se3 import SE3

class Camera:
    """
    Camera class
    TODO:
        - find transformation between mazes coords and world coords
    """
    def __init__(self):
        pass

    # === Public ===
    def get_maze_pose(self) -> SE3:
        """
        Returns a transformation between maze system
        and world system
        """
        pass

    def get_maze_ang_vel(self) -> float:
        """
        Returns an angular velocity of the maze
        """
        pass

    # === Private ===
    # example (private methods are denoted by _name)
    def _find_markers(self):
        pass
    
    def _update_frame(self):
        pass