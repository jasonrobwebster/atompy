"""AtomicState class"""

__all__ = [
    'AtomicState'
]

class AtomicState():
    """Defines an atomic energy level.

    Parameters
    ==========

    E : Number, Symbol
        The energy eignestate of the atomic level.

    level_ket : AtomicJzKet, JzKet
        The atomic ket |n, s, l, j, m_j> that defines the level.

    label : String
        A label for this level.

    atomic_label : String, Optional
        The corresponding atomic label n^(2s+1)L_J.
    """

    def __init__(self, E, ket, label, atomic_label=None):
        # We're assuming that all error handling
        # has been done outside this class.
        self.E = E
        self.ket = ket
        self.label = label
        # TODO: bring the label and atomic_label functionality here.
        self.atomic_label = atomic_label

    def __repr__(self, sep='  '):
        # TODO: Think of a better repr method
        out = 'label={0}' + sep + 'atomic_label={1}' + sep + 'energy={2}' + sep + 'ket=' + str({3})
        out = out.format(
            self.label,
            self.atomic_label,
            self.E,
            self.ket
        )
        return out
