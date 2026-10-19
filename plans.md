# Plan for semstral project

> So first of all our code will be **OOP** so everyone can just use 
> each others classes with no need to know how it works.  

Main classes are:

## Camera

For this project we would need to work with camera.
Why? So we can find a pose of the maze.

### Milestones:
- [ ] AuCro detection
- [ ] Angular speed

### Public:
```python
class Camera:

    # === Public ===
    def get_maze_pose(self):
        """
        Return a mazes pose
        x, y, z, theta
        Considering that the maze is rotating only by 1 axis
        """
        pass

    def get_maze_ang_vel(self):
        """
        I guess compares several frames to determine mazes angular velocity.
        Also direction of rotation.
        """
        pass

    def get_maze_type(self):
        """
        Not sure but different mazes could have different AuCro markers.
        So as a gimmick we could make a maze recognition
        """
```
  
## Planning

As we have maze's pose and we know its geometry we need to somehow plan our trajectory.  
> *Guess we could also get maze type using our camera*

The idea is that we could construct a parameterized curve. So then we have an inner var *"progress"* meaning 
how far did we go by our trajectory.

> Like just $\mathbf{F}(p)$ which returns a point $\begin{bmatrix}x & y & z\end{bmatrix}^T$

Using it we could then get its first and second order derivatives to get normal vector *(to rotate instrument the right way)* and curvature *(for more robust speed control)*

Also we need to plan with angular velocity in mind *(compensate it)*

### Milestones:
- [ ] Parametrization
- [ ] Tangent vectors
- [ ] Curvature

### Public:
``` python
class Planer():

    # === Public === 
    def update_progress(self, hoop_pose) -> None:
        """Updates current progress based on hoop position in maze coordinates"""
        pass

    def get_next_point(self) -> Point:
        """
        Returns the next waypoint for hoop to pass
        
        output:
            (x,y,z), (t_x, t_y, t_z)
        """
        pass

    def get_curve(self) -> double:
        """
        Returns a curvature of a trajectory
        """
        pass
```

## Robot

The last but not least we need an executor.

> Like a driver of a robot it gets where we want it to be and where it is, possibly how we want it to get to the point

### Milestones
- [ ] Make robot to behave
- [ ] IK
- [ ] FK
- [ ] Velocity controls

### Public
```python
class Robot:

    # === Public ===
    def get_hoop_pose(self):
        """
        Returns a center of a hoop and its normal vector

        Guess use a transformation
        """
        pass

    def go_to_point(self, point) -> bool:
        """
        Essentialy a check if we are where we want to be
        point :
            x, y, z, r, p, y
        """
        pass
```

## End
All of that will be tangled up together in main