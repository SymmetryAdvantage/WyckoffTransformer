"""Composition-level estimation of ``f*(X)``, the floor under a chemical formula.

``f*(X)`` is the lowest formation energy any structure with reduced formula ``X``
can have. It is never observed. Every archive entry is a structure that exists,
so its energy is an *upper bound* on the floor, and how loose that bound is
depends on how hard somebody looked -- which is a property of the database's
authors, not of the chemistry.

This package estimates the floor from bounds, by fitting the bound as a bound:
the censored likelihood in :mod:`wyckoff_transformer.censored`, already written
for ``min(E | gene)``, applied one level up at the composition. What the
composition level adds is that the looseness is *measurable*: LeMat-Bulk records
which database each entry came from, and Materials Project records which entries
descend from an experimentally observed structure. Those provenance channels
feed the excess-scale head; the chemistry alone feeds the location head, so the
estimate of the floor cannot learn how fashionable a chemical system is.

See ``docs/composition_screening.md``.
"""
