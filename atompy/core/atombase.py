from __future__ import print_function

from . import sympify

class AtomBase(type):
    """
    Most basic class of the atom.
    """

    def __init__(cls, *args, **kwargs):
        cls.args = args
        cls.kwargs = kwargs
    