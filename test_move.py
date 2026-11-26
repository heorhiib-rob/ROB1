from ctu_crs import CRS93
import cv2
import numpy as np
from se3 import SE3
from so3 import SO3


robot = CRS93()

robot.initialize(home=False)
robot.soft_home()

hooph = SE3(translation=np.array([-0.135, 0, 0]))
hoop = hooph.homogeneous()
pose = robot.fk(robot.get_q())
#for i,p in enumerate(pose):
    #print(f"Joint {i+1} at : {p}")

target_pose = SE3(np.array([
    0.4,
    0,
    0.09
]), rotation=SO3([[-1, 0, 0],[0, 1, 0], [0, 0, -1]])).homogeneous()

q_t = None
for target_pose in [target_pose @hooph.inverse().homogeneous(),target_pose @ SE3(translation=[0,0,0], rotation=SO3(rotation_matrix=[[-1, 0, 0], [0, -1, 0], [0, 0, 1]])).homogeneous() @hooph.inverse().homogeneous()]:

        
    print(target_pose)
    q_ik = robot.ik(target_pose)

    q_c = robot.get_q()

    print(q_ik)

    q_t = None
    best_n = float('inf')

    for q in q_ik:
        if not robot.in_limits(q):
            continue

        score = np.linalg.norm(q - q_c)
        if score < best_n:
            best_n = score
            q_t = q
    if q_t is not None:
        break
    
print(q_t)
#q_t = q_c + np.array([0, -np.pi/6, -np.pi/4, 0, -np.pi/12, 0 ])
if q_t is not None:
    robot.move_to_q(q_t)
    robot.wait_for_motion_stop()

print(robot.fk(robot.get_q())@hoop)
q_robot = robot.get_q()
#print(q_robot)
pose = robot.fk(q_robot)
#for i,p in enumerate(pose):
    #print(f"Joint {i+1} at : {p}")

#qs = robot.ik(pose)
#print(qs)
#cv2.imshow("bla", image)
#cv2.waitKey(10000)

robot.close()


