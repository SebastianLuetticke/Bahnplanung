from dm_control import mjcf
import numpy as np
import os

_SAUSAGE_XML = os.path.join(
    os.path.dirname(__file__),
    '../assets/sausage.xml',
)

class Sausage():
    """
    A class representing a sausage in a simulation environment.
    """
    
    def __init__(self):
        """
        Initialize the sausage object.
        """
        self._mjcf_root = mjcf.from_path(_SAUSAGE_XML)
        
        self._joints = self.mjcf_model.find_all('joint')
        self._geoms = self.mjcf_model.find_all('geom')
        self._sites_by_name = {
            site.name: site
            for site in self.mjcf_model.find_all('site')
        }
        self._site = self.mjcf_model.find('site', 'sausage_site_middle')
        
    @property
    def sites(self):
        """Dictionary of site elements belonging to the sausage."""
        return self._sites_by_name
        
    @property
    def joints(self):
        """List of joint elements belonging to the sausage."""
        return self._joints
    
    @property
    def geom(self):
        """List of geom elements belonging to the sausage."""
        return self._geoms

    @property
    def mjcf_model(self):
        """Returns the sausage's mjcf model."""
        return self._mjcf_root