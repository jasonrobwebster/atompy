"""Base Atomic class handling."""

from __future__ import print_function

from sympy.core.compatibility import range
from atompy.core import (sympify, symbols, Abs, sqrt, S, clebsch_gordan,
                         wigner_6j, m_values, hbar, I, qapply, couple_tensor, 
                         AtomicState, DoubleBar, SphericalTensor, Function, 
                         Sum, Add, Mul, CG, Dummy, Eq)
from atompy.multilevel import AtomicJzKet
from atompy.functions import weak_zeeman, transition_strength, operator_commutator, lindblad_superop

__all__ = [
    'Atom'
]


#-----------------------------------------------------------------------------
# Helper Functions
#-----------------------------------------------------------------------------


def f_values(J, I):
    """Calculates how many F values exist between F=Abs(J-I) to F=J+I"""
    f_min = Abs(J-I)
    f_max = J + I
    f_diff = f_max - f_min
    size = 2*f_diff + 1
    if not size.is_Integer or size < 0:
        raise ValueError('size should be an integer, got J, I, size: %s, %s, %s' % (J, I, size))
    return size, [f for f in range(f_min, f_max + 1)]

t = symbols('t')


#-----------------------------------------------------------------------------
# Atom Class
#-----------------------------------------------------------------------------


class Atom():
    """Define a new atom.

    Parameters
    ==========

    name : String
        The name of the atom (Rb, H, Hydrogen, etc).

    spin : Number, Symbol
        The spin of the nucleus.
        Default is 0.

    mu : Number, Symbol
        The magnetic moment of the atom.
        Default is 0.

    mass : Number, Symbol
        The mass of the atom.

    b : Number, Symbol
        The magnetic field that the atom is subjected to.

    isotope : Int, Symbol
        The isotope number of the atom.

    protons : Int, Symbol
        The number of protons within the atom.

    neutrons : Int, Symbol
        The number of neutrons within the atom.

    electrons : Int, Symbol
        The number of electrons within the atom.
    """

    #-------------------------------------------------------------------------
    # Variables
    #-------------------------------------------------------------------------

    _atomic_labels = set()
    _labels = set()

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def name(self):
        """The name (or symbol) of the atom."""
        return self.kwargs.get('name', '')

    @property
    def spin(self):
        """The internal spin of the nucleus."""
        return self.kwargs.get('spin', S.Zero)

    @property
    def mu(self):
        """The internal nuclear magnetic moment of the atom."""
        return self.kwargs.get('mu', S.Zero)

    @property
    def mass(self):
        """The mass of the atom."""
        return self.kwargs.get('mass')

    @property
    def b(self):
        return self.kwargs.get('b', S.Zero)

    @property
    def isotope(self):
        """The nuclear isotope of the atom."""
        return self.kwargs.get('isotope')

    @property
    def protons(self):
        """The number of protons."""
        return self.kwargs.get('protons')

    @property
    def neutrons(self):
        """The number of neutrons."""
        return self.kwargs.get('neutrons')

    @property
    def electrons(self):
        """The number of electrons."""
        return self.kwargs.get('electrons')

    @property
    def levels(self):
        """The number of atomic levels."""
        return self._levels

    @property
    def labels(self):
        """Returns a copy of the unique level labels in the atom."""
        return self._labels.copy()

    @property
    def atomic_labels(self):
        """Returns a copy of the unique atomic labels in the atom."""
        return self._atomic_labels.copy()

    @property
    def rho(self):
        """The atomic density matrix."""
        return self._rho

    @property
    def hamiltonian(self):
        """The hamiltonian for the atom."""
        return self._hamiltonian

    #-------------------------------------------------------------------------
    # Class Methods
    #-------------------------------------------------------------------------

    @classmethod
    def hartree(cls):
        """The Hartree method for deterimining the energy levels and states."""
        pass

    @classmethod
    def hartree_fock(cls):
        """The Hartree-Fock method for deterimining the energy levels and states."""
        pass

    #-------------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------------

    def __init__(self, **kwargs):
        # sympify the kwargs
        for key in kwargs:
            if key == 'name':
                kwargs[key] = str(kwargs[key])
            else:
                kwargs[key] = sympify(kwargs[key])

        self.kwargs = kwargs
        self._levels_list = [] # a list of the added levels
        self._levels = 0
        self._rho = 0
        self._hamiltonian = 0
        self._rho_list = [] # a list tuples describing the added rho functions (rho_12, |1>, |2>)

    def __repr__(self):
        if self.name != '':
            # init the output
            out = []

            # append the isotope number
            if self.isotope is not None:
                out.append('{}^{' + str(self.isotope) + '}')

            # append the neutron number
            if self.neutrons is not None:
                out.append('{}_{' + str(self.neutrons) + '}')

            # append the name
            out.append(self.name)

            # append the charge
            if self.electrons is not None and self.protons is not None:
                charge = self.protons - self.electrons
                if charge > 0:
                    out.append('^{' + str(abs(charge)) + '+}')
                elif charge < 0:
                    out.append('^{' + str(abs(charge)) + '-}')

            # join everything
            out = ''.join(out)
            return out
        else:
            out = 'Atom({0}={1}, {2}={3}, {4}={5})'.format(
                'levels', self.levels,
                'spin', self.spin,
                'mu', self.mu
            )
            return out

    #-------------------------------------------------------------------------
    # Operations
    #-------------------------------------------------------------------------

    def add_level(self, energy, n, s, l, j, *args, **kwargs):
        """Add an atomic energy level to the atom. 
        Returns an AtomicLevel class representing the state.
        All parameters are passed as keyword arguments.

        Parameters
        ==========

        energy : Number, Symbol
            The atomic energy level before a zeeman shift.

        n : Int, Symbol
            The atomic energy level number, or principle number.

        s: Number, Symbol
            The total electron spin of the atomic level.

        l : Int, String, Symbol
            The atomic level's L total angular momentum.
            Accepts an integer or an atomic label ('S', 'P', etc).

        j : Number, Symbol
            The atomic level's J total angular momentum.
            If the spin of the atom is non zero, this becomes a coupling
            paramter for l and s.

        f : Number, Symbol
            The atomic level's F total angular momentum.
            This argument is not required if the atomic spin is zero.

        m : Number, Symbol
            The atomic level's eigenstate of the Jz operator. If the atomic
            spin is non zero, this is the eigenstate of the Fz operator.

        label: String, Optional
            The unique label for the state.
            This will be used in the notation of the density matrix.
            If none is given, the label will divert to a numbered index.


        Additional kwargs are passed to the weak_zeeman function. See weak_zeeman for usage.

        Examples
        ========



        See Also
        ========

        The weak_zeeman function.
        """
        # TODO: Make examples

        # TODO: Rework the input? 
        # Make it so that we don't assume the user knows nothing
        # of sympy. Allow them to input coupled or decoupled states.

        # TODO: Create automatic labels for 'E'
        E = sympify(energy)
        n = sympify(n)
        s = sympify(s)
        #l = l
        j = sympify(j)
        i = self.spin

        if len(args) > 2:
            raise TypeError('Too many arguments')

        if i != 0:
            if len(args) == 2:
                f = sympify(args[0])
                m = sympify(args[1])
            elif len(args) <= 1:
                if len(args) == 0:
                    if kwargs.get('f') is not None:
                        f = sympify(kwargs.pop('f'))
                    else:
                        raise TypeError('Missing required value f.')
                if kwargs.get('m') is not None:
                    m = sympify(kwargs.pop('m'))
                else:
                    raise TypeError('Missing required value m.')
        elif i == 0:
            if len(args) >= 1:
                m = sympify(args[0])
            elif len(args) == 0:
                if kwargs.get('m') is not None:
                    m = sympify(kwargs.pop('m'))
                else:
                    raise TypeError('Missing required value m.')

        if E.is_Number:
            if not E.is_real:
                raise ValueError('The energy of the level must be real, got %s' % E)

        # convert l to an atomic label and error check it
        l_labels = ['S', 'P', 'D', 'F']
        if isinstance(l, str):
            if str(l) in l_labels:
                l = sympify(l_labels.index(l))
            elif ord(l) <= ord('Z') and ord(l) > ord('F') and len(l) == 1:
                l = sympify(ord(l))
            else:
                raise ValueError('l should be a label, integer, or half integer, got: %s' % l)
        else:
            l = sympify(l)

        # make the level ket, also  performs error checking
        i = self.spin
        if i == 0:
            level_ket = AtomicJzKet(n, j, m, (s, l))
        else:
            if f.is_number and j.is_number and i.is_number:
                if abs(j-i) > f or abs(j+i) < f:
                    raise ValueError(
                        'f must be in the bounds |j-i|<=f<=|j+i|, got j, i, f: %s, %s, %s' % (j, i, f)
                    )
            level_ket = AtomicJzKet(n, f, m, (s, l, i), [(1, 2, j), (1, 3, f)])

        # calculate the energy shift due to a weak magnetic field
        E_shift = weak_zeeman(level_ket, self.b, **kwargs)
        E_level = E + E_shift

        # get the label necessary for the atomic label
        if l.is_Number:
            if l < 4:
                l_label = l_labels[l]
            elif l >= 4:
                l_label = chr(ord('F') + l - 4)

        # The atomic label
        # This differes from the user defined label,
        # which is used to make the notation of the density
        # matrix more convenient
        if i == 0:
            atomic_label = str(n) + l_label + '_' + str(j) + '(m_j=' + str(m) + ')'
        else:
            atomic_label = str(n) + l_label + '_' + str(j) + '(F=' + str(f) + ',m_f=' + str(m) + ')'
        if atomic_label not in self._atomic_labels:
            self._atomic_labels.add(atomic_label)
        else:
            raise ValueError(
                'The level you are adding already exists in this atom: %s' % atomic_label
            )

        # get the label
        if kwargs.get('label') is not None:
            label = kwargs.pop('label')
        else:
            label = None
        # make the label if none exists
        if label is None:
            label = 0
            while str(label) in self._labels:
                label += 1
        label = str(label)

        # check its uniqueness
        if label not in self._labels:
            self._labels.add(label)
        else:
            raise ValueError(
                'The given label already exists in this atom.\n'
                + 'Existing labels: %s, given: %s' % (self.labels, label)
                )

        # add to the hamiltonian
        level_op = level_ket * level_ket.dual
        self._hamiltonian += E * level_op

        # add to the density matrix
        rho_label = 'rho_%s%s' %(label, label)
        rho_label = Function(rho_label)
        self._rho += rho_label(t) * level_op 
        self._rho_list.append((rho_label(t), level_ket, level_ket))
        for state in self._levels_list:
            # add upper part
            old_ket = state.ket
            old_label = state.label
            rho_label = 'rho_%s%s' %(label, old_label)
            rho_label = Function(rho_label)
            rho_op = level_ket * old_ket.dual
            self._rho += rho_label(t) * rho_op
            self._rho_list.append((rho_label(t), level_ket, old_ket))
            # add lower part
            rho_label = 'rho_%s%s' %(old_label, label)
            rho_label = Function(rho_label)
            rho_op = old_ket * level_ket.dual
            self._rho += rho_label(t) * rho_op
            self._rho_list.append((rho_label(t), old_ket, level_ket))

        # add to the level
        new_level = AtomicState(E_level, level_ket, label, atomic_label)
        self._levels_list.append(new_level)
        self._levels += 1

        return new_level
    
    def add_m_sublevels(self, energy, n, s, l, j, *args, **kwargs):
        """Adds all m sublevels for a given atomic state.
        Returns a list of the added sublevels.

        Parameters
        ==========

        energy : Number, Symbol
            The atomic energy level before zeeman splitting.

        n : Int, Symbol
            The atomic energy level number, or principle number.

        s: Number, Symbol
            The total electron spin of the atomic level.

        l : Int, String, Symbol
            The atomic level's L total angular momentum.
            Accepts an integer or an atomic label ('S', 'P', etc).

        j : Number, Symbol
            The atomic level's J total angular momentum.
            If the spin of the atom is non zero, this becomes a coupling
            paramter for l and s.

        f : Number, Symbol
            The atomic level's F total angular momentum.
            This argument is not required if the atomic spin is zero.

        labels : List
            List of strings that provide a unique label for each of the added
            sublevels. Should be of length 2*f+1, or 2*j+1 if the atomic spin
            is zero. The labels in this list should correspond to the order of
            m values given by m = -f, m = -f+1, ..., m = f.


        Additional kwargs are passed to the add_level function. See add_level for usage.

        Examples
        ========



        See Also
        ========

        The add_level function.
        """

        E = sympify(energy)
        n = sympify(n)
        s = sympify(s)
        #l = l
        j = sympify(j)
        i = self.spin

        if kwargs.get('labels') is not None:
            labels = kwargs.pop('labels')
        else:
            labels = None
        
        if i != 0:
            if len(args) == 1:
                f = sympify(args[0])
            elif len(args) == 0:
                if kwargs.get('f') is not None:
                    f = sympify(kwargs.pop('f'))
                else:
                    raise TypeError('Missing required value f.')
            else:
                raise TypeError('Too many arguments')
        else:
            f = j

        # convert l to an atomic label and error check it
        l_labels = ['S', 'P', 'D', 'F']
        if isinstance(l, str):
            if str(l) in l_labels:
                l = sympify(l_labels.index(l))
            elif ord(l) <= ord('Z') and ord(l) > ord('F') and len(l) == 1:
                l = sympify(ord(l))
            else:
                raise ValueError('l should be a label, integer, or half integer, got: %s' % l)
        else:
            l = sympify(l)

        m_vals = [-f + i for i in range(2*f+1)]

        if labels is not None:
            labels = list(labels)
            if len(labels) != 2*f+1:
                if i == 0:
                    raise ValueError(
                        'Length of the labels should be of length 2j+1, got %s' % len(labels)
                    )
                else:
                    raise ValueError(
                        'Length of the labels should be of length 2f+1, got %s' % len(labels)
                    )

        out = []
        for index, m in enumerate(m_vals):
            if i == 0:
                args = tuple([m])
            else:
                args = (f, m)
            # TODO: find a better way to do this
            # must be some python trickery that allows conditional keywords based
            # on whether or not they are None
            if labels is not None:
                state = self.add_level(E, n, s, l, j, *args, label=labels[index])
            else:
                state = self.add_level(E, n, s, l, j, *args)
            out.append(state)

        return out

    def interaction_field(self, dipole, light_field, **kwargs):
        """Returns the hamiltonian due to atomic field interactions.
        Returns the expecation value of -d.E between all ground and excited states.
        Equations are derived from [1] and [2].

        Parameters
        ==========

        dipole: SphericalTensor
            Represents the dipole operator d. Can also be an instance of add.

        light_field: SphericalTensor
            A spherical tensor representitive of the light field.
            Typically something like T^(0)_0 for a plane wave, or
            T^(1)_1 for light with a OAM [2].


        Passes kwargs to transition_strength, see transition_strength for more details.

        References
        ==========
        
        .. [1] Steck, D.A., 2007. Quantum and atom optics. p. 360-375
            http://atomoptics-nas.uoregon.edu/~dsteck/teaching/quantum-optics/quantum-optics-notes.pdf

        .. [2] Schmiegelow, C.T. and Schmidt-Kaler, F., 2012. 
            Light with orbital angular momentum interacting with trapped ions.
        """
        
        # TODO: Build a more robust way to build the interaction hamiltonian
        # Specifically, follow the paper [2] by Schimdt-Kaler.
        # For now just interact directly with one spherical tensor.

        # TODO: Add support for d, E as vectors.

        # TODO: Fix janky way of handling Add and Mul classes 
        out = 0
        
        if isinstance(light_field, Add):
            for arg in light_field.args:
                out += self.interaction_field(dipole, arg)

        if isinstance(light_field, Mul):
            coeff, field = light_field.args
            assert isinstance(coeff, SphericalTensor) is False
            out += coeff * self.interaction_field(dipole, light_field)

        if isinstance(dipole, Add):
            for arg in dipole.args:
                out += self.interaction_field(arg, light_field)

        if isinstance(dipole, Mul):
            coeff, dip = dipole.args
            assert isinstance(coeff, SphericalTensor) is False
            out += coeff * self.interaction_field(dip, light_field)
        
        tensor = couple_tensor(light_field, dipole)

        # gives H_AF=-d.E = -(One) * d.E * (One) where One=sum(|n, j, m><n, j, m|) for all n, j, m, etc
        # Works out to |n, j, m><n, j, m|tensor|n', j', m'><n', j', m'| + c.c. for all the levels
        # The c.c. is |n', j', m'><n', j', m'|tensor|n, j, m><n, j, m| but <n', j', m'|tensor|n, j, m>
        # = <n', j'||tensor||n, j><j', m'|j, m; k, q> = (-1)^q<n, j||tensor||n',j'><j, m| j', m', k, q>
        # and the function transition_strength(g, e, tensor) returns <n, j||tensor||n',j'><j, m| j', m', k, q>
        for ground_state in self._levels_list:
            E_g = ground_state.E
            for excited_state in self._levels_list:
                E_e = excited_state.E
                if E_e > E_g:
                    g_ket = ground_state.ket
                    e_ket = excited_state.ket
                    q = g_ket.m - e_ket.m
                    af_op =  g_ket * e_ket.dual
                    af_op_dual = e_ket * g_ket.dual

                    result = transition_strength(ground_state,
                                                 excited_state,
                                                 tensor,
                                                 **kwargs)
                    result *= af_op

                    result_dual = transition_strength(ground_state,
                                                      excited_state,
                                                      tensor,
                                                      flip_q=True,
                                                      **kwargs)
                    result_dual *= (-1)**q * af_op_dual

                    out += -(result + result_dual) #sign is from -d.E
                    
        return out

    def master_equation(self, pol, light_field, steady=False, **kwargs):
        """Returns a system of equations derived from the master equation [1].

        Parameters
        ==========

        pol : Number
            The polarization of light that is driving atomic transitions. Either -1, 0, or 1.

        light_field: SphericalTensor
            A spherical tensor, or addition of spherical tensors, that describe a light field.

        steady : Boolean, Optional
            Whether the returned master equations should be the steady state master equations.
            Defaults to False.Abs

        References
        ==========

        .. [1] Steck, D.A., 2007. Quantum and atom optics. p. 375
        http://atomoptics-nas.uoregon.edu/~dsteck/teaching/quantum-optics/quantum-optics-notes.pdf
        """

        
        # TODO: Make dipole a vector that can dot into the light_field

        dipole = SphericalTensor(1, -pol)
        # for brevity
        h = self._hamiltonian + self.interaction_field(dipole, light_field, **kwargs)

        result = -I/hbar * operator_commutator(h, self.rho)

        # add the lindblad superops
        for ground_state in self._levels_list:
            E_g = ground_state.E
            for excited_state in self._levels_list:
                E_e = excited_state.E
                if E_e > E_g:
                    lower_op = ground_state.ket * excited_state.ket.dual
                    g_label = ground_state.label
                    e_label = excited_state.label
                    gamma = sympify('Gamma_%s%s' %(g_label, e_label))
                    result += gamma * lindblad_superop(lower_op, self.rho)

        # return a system of equations
        sys = []
        for ground_state in self._levels_list:
            for excited_state in self._levels_list:
                g_label = ground_state.label
                e_label = excited_state.label
                rho = Function('rho_%s%s' %(g_label, e_label))
                #take <g|result|e> to get d/dt rho_ge
                rhs = qapply(result * excited_state.ket)
                rhs = qapply(ground_state.ket.dual * rhs)
                if steady:
                    sys.append(Eq(0, rhs))
                else:
                    sys.append(Eq(rho(t).diff(t), rhs))
        
        return sys


    # TODO: Add method to calculate spin orbit coupling
    # TODO: Add support for adding all j and m_j sublevels for a given L and S
    # TODO: Add a method to calculate hyperfine splitting
    # TODO: Add a way to decompose an arbitraryily defined function into spherical tensor components.
