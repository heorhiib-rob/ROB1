#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO3:
    """This class represents an SO3 rotations internally represented by rotation
    matrix."""

    def __init__(self, rotation_matrix: ArrayLike | None = None) -> None:
        """Creates a rotation transformation from rot_vector."""
        super().__init__()
        self.rot: np.ndarray = (
            np.asarray(rotation_matrix) if rotation_matrix is not None else np.eye(3)
        )

    @staticmethod
    def exp(rot_vector: ArrayLike) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""
        v = np.asarray(rot_vector)
        assert v.shape == (3,)
        t = SO3()
        # DONE HW01: implement Rodrigues' formula, t.rot = ...
        skew_mat = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        theta = np.linalg.norm(v)

        if theta == 0:
            t.rot = np.eye(3)
        else:
            t.rot = (np.eye(3) 
                     + (np.sin(theta)/theta)*skew_mat 
                     + ((1 - np.cos(theta))/theta**2) * skew_mat @ skew_mat)
        return t

    def log(self) -> np.ndarray:
        """Compute rotation vector from this SO3"""
        # DONE HW01: implement computation of rotation vector from this SO3
        v = np.zeros(3)
        if np.allclose(self.rot, np.eye(3)):
                return v
        trR = np.trace(self.rot)
        theta = np.arccos((trR - 1)/2)

        omega_r = (1/(2*np.sin(theta)) 
                   * np.array([
                self.rot[2, 1] - self.rot[1, 2],
                self.rot[0, 2] - self.rot[2, 0],
                self.rot[1, 0] - self.rot[0, 1]
                   ]))

        return theta * omega_r


    def __mul__(self, other: SO3) -> SO3:
        """Compose two rotations, i.e., self * other"""
        res = SO3()
        res.rot = self.rot @ other.rot
        return res

    def inverse(self) -> SO3:
        """Return inverse of the transformation."""
        # DONE: HW01: implement inverse, do not use np.linalg.inverse()
        res = SO3()
        res.rot = self.rot.T
        return res

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (3,)
        return self.rot @ v

    def __eq__(self, other: SO3) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    @staticmethod
    def rx(angle: float) -> SO3:
        """Return rotation matrix around x axis."""
        # DONE: HW1opt: implement rx
        s = np.sin(angle)
        c = np.cos(angle)
        res = SO3()
        res.rot = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0,  s, c]
        ])
        return res

    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y axis."""
        # DONE: HW1opt: implement ry
        s = np.sin(angle)
        c = np.cos(angle)
        res = SO3()
        res.rot = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
        return res

    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z axis."""
        # DONE: HW1opt: implement rz
        s = np.sin(angle)
        c = np.cos(angle)
        res = SO3()
        res.rot = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        return res

    @staticmethod
    def from_quaternion(q: ArrayLike) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""
        # DONE: HW1opt: implement from quaternion
        q_w = q[3]
        ac = 2 * np.arccos(q_w)
        q_xyz = q[:3]

        res = SO3().exp(ac*(q_xyz/np.linalg.norm(q_xyz)))
        return res

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""
        # DONE: HW1opt: implement to quaternion
        omega = self.log()
        theta = np.linalg.norm(omega)

        if theta < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0])

        omega_r = omega/theta
        q_w = np.array([np.cos(theta/2)])
        q_xyz = omega_r*np.sin(theta/2)

        return np.concatenate([q_xyz,q_w])

    @staticmethod
    def from_angle_axis(angle: float, axis: ArrayLike) -> SO3:
        """Compute rotation from angle axis representation."""
        # DONE: HW1opt: implement from angle axis
        w = angle * axis
        res = SO3().exp(w)
        return res

    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""
        # TODO: HW1opt: implement to angle axis
        w = self.log()
        angle = np.linalg.norm(w)
        if angle < 1e-12:
            return 0

        axis = w/angle
        return (angle, axis)

    @staticmethod
    def from_euler_angles(angles: ArrayLike, seq: list[str]) -> SO3:
        """Compute rotation from euler angles defined by a given sequence.
        angles: is a three-dimensional array of angles
        seq: is a list of axis around which angles rotate, e.g. 'xyz', 'xzx', etc.
        """
        # TODO: HW1opt: implement from euler angles
        res = SO3()
        for i in range(len(seq)):
            axis = seq[i]
            angle = angles[i]
            match axis:
                case 'x':
                    res = res*SO3().rx(angle)
                case 'y':
                    res = res*SO3().ry(angle)
                case 'z':
                    res = res*SO3().rz(angle)
        return res

    def __hash__(self):
        return id(self)
