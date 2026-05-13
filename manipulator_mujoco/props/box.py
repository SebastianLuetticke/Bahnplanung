from dm_control import mjcf
import numpy as np

class Box(object):
    """
    A class representing a box object in a simulation environment.
    """

    def __init__(self, colliders_on=None, collidier_rgba=[1, 0.5, 0, 1], **kwargs):
        """
        Initialize the box object.

        Args:
            colliders_on: List of booleans to determine which sides are collidable [left, right, front, back, bottom, top].
            collidier_rgba: Color of the collider sides.
            **kwargs: Additional keyword arguments for configuring the box geom.
        """
        
        name = kwargs.pop("name", "unnamed")
        
        self._mjcf_model = mjcf.RootElement(model=f"{name}_model")
        
        self._body = self._mjcf_model.worldbody.add(
            "body",
            name=f"{name}_body",
            pos=kwargs.pop("pos", [0, 0 ,0]),
            quat=kwargs.pop("quat", [1, 0, 0, 0]),
        )

        self._geom = self._body.add(
            "geom",
            name=f"{name}_geom",
            mass="1000",
            **kwargs
        )
        
        if colliders_on is None:
            colliders_on = [False]*6
        self._collider_thickness = 0.0005
        self._collider_geoms = []
        
        for i in range(6):
            if colliders_on[i]:
                dimension_index = i // 2   # two colliders for each dimension
                collider_pos = [0, 0, 0]
                if i % 2 == 0:
                    collider_pos[dimension_index] = -kwargs["size"][dimension_index] - self._collider_thickness
                else:
                    collider_pos[dimension_index] = kwargs["size"][dimension_index] + self._collider_thickness
                collider_size = kwargs["size"].copy()
                collider_size[dimension_index] = self._collider_thickness
                self._collider_geoms.append(self._body.add(
                    "geom",
                    name=f"{name}_collision{i}",
                    type="box",
                    pos=collider_pos,
                    size=collider_size,
                    rgba=collidier_rgba
                    ))

    @property
    def geom(self):
        """Returns the geom of the box."""
        return self._geom
    
    @property
    def mjcf_model(self):
        """Returns the mjcf model of the box."""
        return self._mjcf_model
