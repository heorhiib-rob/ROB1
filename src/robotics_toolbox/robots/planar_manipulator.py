#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-08-21
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing planar manipulator."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from shapely import MultiPolygon, LineString, MultiLineString

from robotics_toolbox.core import SE2, SE3
from robotics_toolbox.robots.robot_base import RobotBase


class PlanarManipulator(RobotBase):
    def __init__(
        self,
        link_parameters: ArrayLike | None = None,
        structure: list[str] | str | None = None,
        base_pose: SE2 | None = None,
        gripper_length: float = 0.2,
    ) -> None:
        """
        Creates a planar manipulator composed by rotational and prismatic joints.

        The manipulator kinematics is defined by following kinematics chain:
         T_flange = (T_base) T(q_0) T(q_1) ... T_n(q_n),
        where
         T_i describes the pose of the next link w.r.t. the previous link computed as:
         T_i = R(q_i) Tx(l_i) if joint is revolute,
         T_i = R(l_i) Tx(q_i) if joint is prismatic,
        with
         l_i is taken from @param link_parameters;
         type of joint is defined by the @param structure.

        Args:
            link_parameters: either the lengths of links attached to revolute joints
             in [m] or initial rotation of prismatic joint [rad].
            structure: sequence of joint types, either R or P, [R]*n by default
            base_pose: mounting of the robot, identity by default
            gripper_length: length of the gripper measured from the flange
        """
        super().__init__()
        self.link_parameters: np.ndarray = np.asarray(
            [0.5] * 3 if link_parameters is None else link_parameters
        )
        n = len(self.link_parameters)
        self.base_pose = SE2() if base_pose is None else base_pose
        self.structure = ["R"] * n if structure is None else structure
        assert len(self.structure) == len(self.link_parameters)
        self.gripper_length = gripper_length

        # Robot configuration:
        self.q = np.array([np.pi / 8] * n)
        self.gripper_opening = 0.2

        # Configuration space
        self.q_min = np.array([-np.pi] * n)
        self.q_max = np.array([np.pi] * n)

        # Additional obstacles for collision checking function
        self.obstacles: MultiPolygon | None = None

    @property
    def dof(self):
        """Return number of degrees of freedom."""
        return len(self.q)

    def sample_configuration(self):
        """Sample robot configuration inside the configuration space. Will change
        internal state."""
        return np.random.uniform(self.q_min, self.q_max)

    def set_configuration(self, configuration: np.ndarray):
        """Set configuration of the robot, return self for chaining."""
        self.q = configuration
        return self

    def configuration(self) -> np.ndarray | SE2 | SE3:
        """Get the robot configuration."""
        return self.q

    def flange_pose(self) -> SE2:
        """Return the pose of the flange in the reference frame."""
        # todo HW02: implement fk for the flange
        return self.fk_all_links()[-1]

    def fk_all_links(self) -> list[SE2]:
        """Compute FK for frames that are attached to the links of the robot.
        The first frame is base_frame, the next frames are described in the constructor.
        """
        # todo HW02: implement fk
        frames = []

        T = self.base_pose
        frames.append(T)

        for idx, joint_type in enumerate(self.structure):
            T_i = SE2()
            if joint_type == "R":
                angle = self.q[idx]
                dist = self.link_parameters[idx]
            elif joint_type == "P":
                angle = self.link_parameters[idx]
                dist = self.q[idx]

            t = np.array([
                dist * np.cos(angle),
                dist * np.sin(angle)
            ])

            T_i = SE2(
                translation=t,
                rotation=angle
            )

            T = T * T_i
            frames.append(T)

        return frames

    def _gripper_lines(self, flange: SE2):
        """Return tuple of lines (start-end point) that are used to plot gripper
        attached to the flange frame."""
        gripper_opening = self.gripper_opening / 2.0
        return (
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([0, +gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([self.gripper_length, -gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, +gripper_opening])).translation,
                (flange * SE2([self.gripper_length, +gripper_opening])).translation,
            ),
        )

    def jacobian(self) -> np.ndarray:
        """Computes jacobian of the manipulator for the given structure and
        configuration."""
        jac = np.zeros((3, len(self.q)))
        # todo: HW04 implement jacobian computation

        frames = self.fk_all_links()
        p_end = frames[-1].translation

        for i, joint_type in enumerate(self.structure):
            if joint_type == "R":
                p_joint = frames[i].translation
                d = p_end - p_joint
                jac[:, i] = [
                    -d[1], d[0], 1
                ]
            elif joint_type == "P":
                T = frames[i + 1]
                direction = (
                    T * SE2([1,0])
                ).translation - T.translation

                jac[:,i] = [
                    direction[0],
                    direction[1],
                    0.0
                ]
        return jac

    def jacobian_finite_difference(self, delta=1e-5) -> np.ndarray:
        jac = np.zeros((3, len(self.q)))
        # todo: HW04 implement jacobian computation

        q_original = np.asarray(self.q, dtype=float).copy()

        for i, _ in enumerate(self.structure):
            q_plus = q_original.copy()
            q_plus[i] += delta

            self.q = q_plus

            f_end = self.flange_pose()
            p_plus = np.array([
                f_end.translation[0],
                f_end.translation[1],
                f_end.rotation.angle
            ])

            q_minus = q_original.copy()
            q_minus[i] -= delta

            self.q = q_minus

            f_end = self.flange_pose()
            p_minus = np.array([
                f_end.translation[0],
                f_end.translation[1],
                f_end.rotation.angle
            ])

            diff = p_plus - p_minus
            diff[2] = (diff[2] + np.pi) % (2*np.pi) - np.pi
            jac[:,i] = diff / (2*delta)

        self.q = q_original

        return jac

    def ik_numerical(
        self,
        flange_pose_desired: SE2,
        max_iterations=1000,
        acceptable_err=1e-4,
    ) -> bool:
        """Compute IK numerically. Value self.q is used as an initial guess and updated
        to solution of IK. Returns True if converged, False otherwise."""
        # todo: HW05 implement numerical IK

        def error(flange_p):
            x_d, y_d = flange_pose_desired.translation
            phi_d = flange_pose_desired.rotation.angle

            x_c, y_c = flange_p.translation
            phi_c = flange_p.rotation.angle

            e_phi = (phi_d - phi_c + np.pi) % (2*np.pi) - np.pi

            return np.array([
                x_d - x_c,
                y_d - y_c,
                e_phi
            ]) 

        for i in range(max_iterations):
            flange = self.flange_pose()

            e = error(flange)
            err_old = np.linalg.norm(e)

            if err_old < acceptable_err:
                return True
            
            J = self.jacobian()
            dq = np.linalg.pinv(J) @ e

            alpha = 1.0
            q_old = self.q.copy()

            accepted = False

            for i in range(30):

                self.q = q_old + alpha * dq

                e_new = error(self.flange_pose())
                err_new = np.linalg.norm(e_new)

                if err_new < err_old:
                    accepted = True
                    break

                alpha *= 0.5

            if not accepted:
                self.q = q_old
                return False
            
        return np.linalg.norm(error(self.flange_pose())) <= acceptable_err

    def ik_analytical(self, flange_pose_desired: SE2) -> list[np.ndarray]:
        """Compute IK analytically, return all solutions for joint limits being
        from -pi to pi for revolute joints -inf to inf for prismatic joints."""
        assert self.structure in (
            "RRR",
            "PRR",
        ), "Only RRR or PRR structure is supported"

        # todo: HW05 implement analytical IK for RRR manipulator
        # todo: HW05 optional implement analytical IK for PRR manipulator
        if self.structure == "RRR":
            # get base pose to be (0,0,0)
            target_local = self.base_pose.inverse() * flange_pose_desired
            
            # single coords of target
            x, y = target_local.translation
            phi = target_local.rotation.angle

            elbow_up = list()
            elbow_down = list()
            # const lens
            l1, l2, l3 = self.link_parameters

            # third joint pose
            x_w = x - l3*np.cos(phi)
            y_w = y - l3*np.sin(phi)

            # * first joint
            d_sq = x_w**2 + y_w**2
            alpha = np.arccos(
                (d_sq + l1**2 - l2**2)/
                (2*l1*np.sqrt(d_sq))
            ) 

            beta = np.atan2(y_w, x_w)

            q1_up = -alpha + beta # elbow up
            q1_down = alpha + beta # elbow down

            elbow_up.append(q1_up)
            elbow_down.append(q1_down)
            # * second joint
            # elbow up
            x_e = l1 * np.cos(q1_up)
            y_e = l1 * np.sin(q1_up)

            theta = np.atan2(y_w - y_e, x_w - x_e) 

            q2_up = theta - q1_up
            elbow_up.append(q2_up)

            # elbow down
            x_e = l1 * np.cos(q1_down)
            y_e = l1 * np.sin(q1_down)

            theta = np.atan2(y_w - y_e, x_w - x_e) 

            q2_down = theta - q1_down
            elbow_down.append(q2_down)

            # * third joint
            # elbow up 
            q3_up = phi - q1_up - q2_up
            elbow_up.append(q3_up)

            # elbow down
            q3_down = phi - q1_down - q2_down
            elbow_down.append(q3_down)

            q_up = np.array(elbow_up)
            q_down = np.array(elbow_down)

            q_up = (q_up + np.pi) % (2*np.pi) - np.pi
            q_down = (q_down + np.pi) % (2*np.pi) - np.pi

            solutions = [q_down, q_up]

        else:
            pose = self.base_pose.inverse() * flange_pose_desired

            x, y = pose.translation
            phi = pose.rotation.angle

            a, l2, l3 = self.link_parameters

            xw = x - l3 * np.cos(phi)
            yw = y - l3 * np.sin(phi)

            t = xw * np.cos(a) + yw * np.sin(a)
            d = -xw * np.sin(a) + yw * np.cos(a)

            if d**2 > l2**2 + 1e-10:
                return []

            h = np.sqrt(max(0, l2**2 - d**2))

            solutions = []

            for sign in [1, -1]:

                q1 = t + sign * h

                q2 = np.arctan2(
                    yw - q1 * np.sin(a),
                    xw - q1 * np.cos(a)
                ) - a

                q3 = phi - a - q2

                q2 = (q2 + np.pi) % (2 * np.pi) - np.pi
                q3 = (q3 + np.pi) % (2 * np.pi) - np.pi

                q = np.array([q1, q2, q3])

                if not any(np.allclose(q, s) for s in solutions):
                    solutions.append(q)

        return solutions

    def in_collision(self) -> bool:
        """Check if robot in its current pose is in collision."""
        frames = self.fk_all_links()
        points = [f.translation for f in frames]
        gripper_lines = self._gripper_lines(frames[-1])

        links = [LineString([a, b]) for a, b in zip(points[:-2], points[1:-1])]
        links += [MultiLineString((*gripper_lines, (points[-2], points[-1])))]
        for i in range(len(links)):
            for j in range(i + 2, len(links)):
                if links[i].intersects(links[j]):
                    return True
        return MultiLineString(
            (*gripper_lines, *zip(points[:-1], points[1:]))
        ).intersects(self.obstacles)
