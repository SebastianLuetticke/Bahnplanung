from dm_control import mjcf
import numpy as np

class Site(object):
    """
    A base class representing a site object in a simulation environment.
    """

    def __init__(self, **kwargs):
        """
        Initialize the site object.

        Args:
            **kwargs: Additional keyword arguments for configuring the site.
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
            "site",
            name=f"{name}_site",
            **kwargs
        )

    @property
    def geom(self):
        """Returns the site's geom."""
        return self._geom
    
    @property
    def mjcf_model(self):
        """Returns the site's mjcf model."""
        return self._mjcf_model
