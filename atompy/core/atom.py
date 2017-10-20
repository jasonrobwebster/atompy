from __future__ import print_function

from .atombase import AtomBase

class Atom(AtomBase):
    """
    The Atom class.

    Used to instantiate classes of atoms. Other atoms will extend from this.
    """

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def isotope(self):
        """The nuclear isotope of the atom."""
        return self.isotope

    @property 
    def protons(self):
        """The number of protons."""
        return self.protons

    @property
    def neutrons(self):
        """The number of neutrons."""
        return self.neutrons
    
    @property
    def electrons(self):
        """The number of electrons."""
        return self.electrons

    @property
    def magnetic_moment(self):
        """The magnetic moment of the atom."""
        return self.magnetic_moment
    
    @property
    def mass(self):
        """The mass of the atom."""
        return self.mass
    
    @property
    def energy_states(self):
        """The energy states of the atom."""
        return self.energy_states

    @property
    def density_matrix(self):
        """The density matrix of the atom."""
        return self.density_matrix
    
    


    
