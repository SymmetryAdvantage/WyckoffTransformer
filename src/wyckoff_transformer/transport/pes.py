"""Potential-energy-surface descriptors of Li mobility from a universal potential.

Two scans, both of a single Li moved over a regular grid in a frozen host:

* **Frozen framework** (Maevskiy et al., Phys. Rev. Research 7, 023167, 2025): one Li of
  the relaxed structure is taken out and put back at every grid point, the rest of the
  structure (other Li included) held fixed.  The grid step along each lattice vector is
  at most 0.25 A and points closer than 1.2 A to another atom are skipped.  The PES is
  measured from its minimum and tiled into a 2x2x2 supercell centred on that minimum.
  From it come the *free volumes* below a threshold, disconnected (E < t anywhere) and
  connected (reachable from the minimum without exceeding t), the *minimal percolation
  energy* (the lowest t at which the reachable region touches the supercell's faces),
  and the combined descriptor
  ``Xi = s(2.00 + log10 FVc(0.5 eV)) * s(1.15 + log10 FVd(0.5 eV))``,
  ``s(x) = 1 / (1 + exp(-10 x))``.  Every Li atom is scanned and the indicator arrays of
  the different atoms, each centred on its own minimum, are combined with a logical OR
  before the volume is taken, as the paper describes.
* **Empty framework** (the ICML 2026 pipeline, its section 3.4): every Li is removed and
  one Li scanned through the empty host.  Its descriptor is the percolation barrier:
  the lowest energy above the PES minimum at which the accessible region percolates in
  1, 2 or 3 dimensions, found by the same bisection on connected components as
  ``bvlain``'s BVSE barriers, so the two numbers are directly comparable.

The connected region at a threshold is the 6-connected component of ``E < t`` that
holds the start; with a 0.25 A grid and 1.2 A exclusion spheres no wall is thin enough
for the choice of connectivity to let a path through it.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

#: Face connectivity for :func:`scipy.ndimage.label`.
FACE_CONNECTIVITY = ndimage.generate_binary_structure(3, 1)


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grid:
    """A regular grid over the unit cell, axes along the lattice vectors."""

    shape: tuple[int, int, int]

    @classmethod
    def for_lattice(cls, matrix: np.ndarray, step: float) -> Grid:
        """The coarsest grid whose step along every lattice vector is at most *step*."""
        lengths = np.linalg.norm(np.asarray(matrix, dtype=float), axis=1)
        return cls(tuple(math.ceil(length / step - 1e-9) for length in lengths))

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def frac_coords(self) -> np.ndarray:
        """Fractional coordinates of every point, C order over (i1, i2, i3)."""
        axes = [np.arange(n) / n for n in self.shape]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def allowed_points(matrix: np.ndarray, atom_frac: np.ndarray, grid: Grid,
                   r_min: float) -> np.ndarray:
    """Which grid points are at least *r_min* from every atom (periodic)."""
    from pymatgen.optimization.neighbors import find_points_in_spheres

    matrix = np.ascontiguousarray(matrix, dtype=float)
    points = np.ascontiguousarray(grid.frac_coords() @ matrix)
    atoms = np.ascontiguousarray(np.asarray(atom_frac, dtype=float) @ matrix)
    centre_index, _, _, _ = find_points_in_spheres(
        atoms, points, r=r_min, pbc=np.array([1, 1, 1], dtype=np.int64), lattice=matrix,
        tol=1e-8)
    allowed = np.ones(grid.size, dtype=bool)
    allowed[np.unique(centre_index)] = False
    return allowed


# --------------------------------------------------------------------------- #
# Energy models
# --------------------------------------------------------------------------- #
class OrbProbeEnergy:
    """Energies of a frozen host plus one probe atom, batched through ORB on a GPU.

    The model and its graph settings are taken from the same calculator the relaxations
    use (:func:`wyckoff_transformer.evaluation.hull_mlips.build_hull_calculator`), so the
    PES and the relaxed structure come from one potential at one precision.
    """

    def __init__(self, mlip: str = "orb_conserv_inf", device: str = "cuda",
                 max_atoms_per_batch: int = 16_000, energy_only: bool = True):
        import torch

        from wyckoff_transformer.evaluation.hull_mlips import build_hull_calculator

        calculator = build_hull_calculator(mlip, device=device)
        self.model = calculator.model
        self.adapter = calculator.adapter
        self.max_num_neighbors = calculator.max_num_neighbors
        self.edge_method = calculator.edge_method
        self.half_supercell = getattr(calculator, "half_supercell", None)
        self.device = torch.device(device)
        self.max_atoms_per_batch = max_atoms_per_batch
        self.energy_only = energy_only
        self.evaluations = 0

    def _energy_only(self, batch):
        """The conservative model's energy without the force and stress backward pass.

        The forward pass of ``ConservativeForcefieldRegressor`` up to its interaction
        energy, run under ``no_grad`` on the edge vectors the adapter already stored.
        Checked against ``predict`` by :meth:`check_energy_only`.
        """
        import torch

        model = self.model
        if not hasattr(model, "energy_name"):
            # A direct model predicts forces with a head, so its predict() has no
            # backward pass to skip.
            with torch.no_grad():
                return model.predict(batch)["energy"]
        if getattr(model, "coulomb_module", None) is not None or "latent_charges" in model.heads:
            raise NotImplementedError("energy-only path does not cover charge heads")
        with torch.no_grad():
            features = model.model(batch)["node_features"]
            head = model.heads[model.energy_name]
            energy = head(features, batch)
            if model.pair_repulsion:
                energy = energy + model.pair_repulsion_fn(batch)["energy"]
            return head.absolute_energy(energy, batch, fp64=True)

    def _batch(self, host, probe_number: int, chunk: np.ndarray):
        """One batched graph of the host plus the probe at each position.

        ``from_ase_atoms_list`` builds every graph in one batched neighbour search; one
        ``from_ase_atoms`` call per configuration costs 15x the forward pass.
        """
        from ase import Atoms

        numbers = np.append(host.numbers, probe_number)
        atoms = [Atoms(numbers=numbers, positions=np.vstack([host.positions, position]),
                       cell=host.cell, pbc=True) for position in chunk]
        return self.adapter.from_ase_atoms_list(
            atoms, max_num_neighbors=self.max_num_neighbors, edge_method=self.edge_method,
            device=self.device).to(self.device)

    def __call__(self, host, probe_number: int, probe_cart: np.ndarray) -> np.ndarray:
        """Total energy of *host* with one atom of *probe_number* at each position."""
        per_batch = max(1, self.max_atoms_per_batch // (len(host) + 1))
        out = np.empty(len(probe_cart))
        for start in range(0, len(probe_cart), per_batch):
            chunk = probe_cart[start:start + per_batch]
            batch = self._batch(host, probe_number, chunk)
            energy = (self._energy_only(batch) if self.energy_only
                      else self.model.predict(batch)["energy"])
            out[start:start + len(chunk)] = energy.detach().double().cpu().numpy()
            self.evaluations += len(chunk)
        return out

    def check_energy_only(self, host, probe_number: int, probe_cart: np.ndarray) -> float:
        """Largest |E_energy_only - E_predict| over the given probes, eV."""
        batch = self._batch(host, probe_number, probe_cart)
        fast = self._energy_only(batch).detach().double().cpu().numpy()
        batch = self._batch(host, probe_number, probe_cart)
        full = self.model.predict(batch)["energy"].detach().double().cpu().numpy()
        return float(np.abs(fast - full).max())


# --------------------------------------------------------------------------- #
# Reading a PES
# --------------------------------------------------------------------------- #
def tile_centred(pes: np.ndarray, start: tuple[int, int, int]) -> np.ndarray:
    """The 2x2x2 supercell of *pes* with *start* at the centre, faces included.

    Entry ``[N1 + c1, N2 + c2, N3 + c3]`` is ``pes[(start + c) mod N]`` for offsets
    ``c_i`` in ``-N_i..N_i``, so the start's 26 periodic replicas lie on the faces.
    """
    index = [(np.arange(-n, n + 1) + s) % n for n, s in zip(pes.shape, start)]
    return pes[np.ix_(*index)]


def reachable(tiled: np.ndarray, threshold: float) -> np.ndarray:
    """The face-connected component of ``tiled < threshold`` holding the centre."""
    centre = tuple(n // 2 for n in tiled.shape)
    below = tiled < threshold
    if not below[centre]:
        return np.zeros_like(below)
    labels, _ = ndimage.label(below, structure=FACE_CONNECTIVITY)
    return labels == labels[centre]


def touches_faces(region: np.ndarray) -> bool:
    return bool(region[0].any() or region[-1].any() or region[:, 0].any()
                or region[:, -1].any() or region[:, :, 0].any() or region[:, :, -1].any())


def minimal_percolation_energy(tiled: np.ndarray, encut: float = 10.0,
                               tol: float = 0.005) -> float:
    """Lowest threshold at which the centre's region reaches the supercell's faces."""
    finite = tiled[np.isfinite(tiled)]
    if not finite.size:
        return math.inf
    low, high = 0.0, min(encut, float(finite.max()) + tol)
    if not touches_faces(reachable(tiled, high)):
        return math.inf
    while high - low > tol:
        middle = 0.5 * (low + high)
        if touches_faces(reachable(tiled, middle)):
            high = middle
        else:
            low = middle
    return high


def sigmoid(x: float, slope: float = 10.0) -> float:
    return 1.0 / (1.0 + math.exp(-slope * x))


def xi_descriptor(fv_connected: float, fv_disconnected: float,
                  connected_threshold: float = -2.00, disconnected_threshold: float = -1.15,
                  slope: float = 10.0) -> float:
    """Maevskiy et al.'s Eq. (4); a zero free volume gives zero."""
    if fv_connected <= 0 or fv_disconnected <= 0:
        return 0.0
    return (sigmoid(-connected_threshold + math.log10(fv_connected), slope)
            * sigmoid(-disconnected_threshold + math.log10(fv_disconnected), slope))


@dataclass
class FrozenFrameworkResult:
    fv_connected: float
    fv_disconnected: float
    mpe: float
    xi: float
    n_mobile: int
    n_evaluations: int
    grid: tuple[int, int, int]
    seconds: float
    per_atom_mpe: list[float] = field(default_factory=list)
    n_scanned: int = 0
    n_orbits: int = 0


def mobile_orbits(structure, mobile_sites: list[int], symprec: float = 0.1
                  ) -> dict[int, tuple[int, np.ndarray]]:
    """Each mobile site -> (its orbit representative, R with R x_rep + t = x_site).

    From spglib at the gene tolerance.  A site that is its own representative maps with
    the identity.  Without symmetry every site represents itself.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    identity = np.eye(3, dtype=int)
    try:
        dataset = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetry_dataset()
    except (TypeError, ValueError):
        dataset = None
    if dataset is None:
        return {i: (i, identity) for i in mobile_sites}
    equivalent = np.asarray(dataset.equivalent_atoms)
    frac = structure.frac_coords
    out = {}
    for site in mobile_sites:
        rep = int(equivalent[site])
        if rep == site:
            out[site] = (site, identity)
            continue
        image = frac[rep] @ np.asarray(dataset.rotations).transpose(0, 2, 1) + \
            np.asarray(dataset.translations)
        delta = image - frac[site]
        delta -= np.round(delta)
        k = int(np.argmin(np.linalg.norm(delta @ structure.lattice.matrix, axis=1)))
        if np.linalg.norm(delta[k] @ structure.lattice.matrix) > 2 * symprec:
            out[site] = (site, identity)       # no operation maps it: scan it directly
        else:
            out[site] = (rep, np.asarray(dataset.rotations[k]))
    return out


def map_relative_mask(mask: np.ndarray, rotation: np.ndarray, shape: tuple[int, int, int]
                      ) -> np.ndarray:
    """A centred-supercell mask of an orbit member from its representative's.

    With x_site = R x_rep + t, the member's PES at an offset f from its minimum is the
    representative's at R^-1 f.  Offsets are resampled to the nearest grid point; ones
    that fall outside the representative's supercell are outside.
    """
    n = np.array(shape)
    axes = [np.arange(-k, k + 1) for k in shape]
    offsets = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    source = np.rint((offsets / n) @ np.linalg.inv(rotation).T * n).astype(int)
    inside = np.all(np.abs(source) <= n, axis=1)
    out = np.zeros(len(offsets), dtype=bool)
    idx = source[inside] + n
    out[inside] = mask[idx[:, 0], idx[:, 1], idx[:, 2]]
    return out.reshape(mask.shape)


def frozen_framework_scan(structure, energy_model, mobile: str = "Li", step: float = 0.25,
                          r_min: float = 1.2, threshold: float = 0.5, encut: float = 10.0,
                          xi_kwargs: dict | None = None, use_symmetry: bool = True,
                          symprec: float = 0.1, max_orbits: int | None = None,
                          seed: int = 0) -> FrozenFrameworkResult:
    """Maevskiy et al.'s frozen-framework descriptors of one relaxed structure.

    Every Li atom contributes to the free volumes.  With *use_symmetry* only one Li per
    symmetry orbit is scanned; the masks of the other members of its orbit are its own,
    carried over by the operation that maps one onto the other
    (:func:`map_relative_mask`).  With *max_orbits*, a structure with more Li orbits
    than that has a seeded random subset of them scanned, and only those orbits'
    atoms contribute: its free volumes are then lower bounds.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    started = time.time()
    matrix = structure.lattice.matrix
    grid = Grid.for_lattice(matrix, step)
    points_cart = grid.frac_coords() @ matrix
    mobile_sites = [i for i, site in enumerate(structure) if site.specie.symbol == mobile]
    probe_number = structure[mobile_sites[0]].specie.Z
    orbits = mobile_orbits(structure, mobile_sites, symprec) if use_symmetry else \
        {i: (i, np.eye(3, dtype=int)) for i in mobile_sites}
    representatives = sorted({rep for rep, _ in orbits.values()})
    n_orbits = len(representatives)
    if max_orbits is not None and n_orbits > max_orbits:
        rng = np.random.default_rng(seed)
        representatives = sorted(rng.choice(representatives, max_orbits, replace=False).tolist())
        mobile_sites = [site for site in mobile_sites if orbits[site][0] in representatives]
    scanned: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    evaluations = 0
    for removed in representatives:
        host = structure.copy()
        host.remove_sites([removed])
        allowed = allowed_points(matrix, host.frac_coords, grid, r_min)
        energies = np.full(grid.size, np.inf)
        if allowed.any():
            atoms = AseAtomsAdaptor.get_atoms(host)
            energies[allowed] = energy_model(atoms, probe_number, points_cart[allowed])
            evaluations += int(allowed.sum())
        pes = (energies - energies[allowed].min()).reshape(grid.shape) if allowed.any() \
            else energies.reshape(grid.shape)
        start = np.unravel_index(int(np.argmin(pes)), grid.shape)
        tiled = tile_centred(pes, start)
        scanned[removed] = (reachable(tiled, threshold), tiled < threshold,
                            minimal_percolation_energy(tiled, encut))
    connected = disconnected = None
    per_atom_mpe = []
    for site in mobile_sites:
        rep, rotation = orbits[site]
        mask_c, mask_d, mpe = scanned[rep]
        if rep != site:
            mask_c = map_relative_mask(mask_c, rotation, grid.shape)
            mask_d = map_relative_mask(mask_d, rotation, grid.shape)
        connected = mask_c if connected is None else connected | mask_c
        disconnected = mask_d if disconnected is None else disconnected | mask_d
        per_atom_mpe.append(mpe)
    fv_c = float(connected.mean()) if connected is not None else 0.0
    fv_d = float(disconnected.mean()) if disconnected is not None else 0.0
    return FrozenFrameworkResult(
        fv_connected=fv_c, fv_disconnected=fv_d,
        mpe=min(per_atom_mpe) if per_atom_mpe else math.inf,
        xi=xi_descriptor(fv_c, fv_d, **(xi_kwargs or {})),
        n_mobile=len(mobile_sites), n_evaluations=evaluations, grid=grid.shape,
        seconds=time.time() - started, per_atom_mpe=per_atom_mpe,
        n_scanned=len(scanned), n_orbits=n_orbits)


# --------------------------------------------------------------------------- #
# Empty framework
# --------------------------------------------------------------------------- #
_SHIFTS = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")).reshape(3, -1).T


def percolation_dimension(labels: np.ndarray, shape: tuple[int, int, int]) -> int:
    """Largest number of unit-cell replicas one component spans in a 2x2x2 tiling.

    2 means 1D percolation, 4 means 2D and 8 means 3D, as in ``bvlain``.  A component
    spans the replica of one of its points if the label at that point's image in
    another cell of the tiling is its own.
    """
    n = np.array(shape)
    values, first = np.unique(labels.ravel(), return_index=True)
    best = 0
    for value, index in zip(values, first):
        if value == 0:
            continue
        base = np.array(np.unravel_index(index, labels.shape)) % n
        replicas = base + _SHIFTS * n
        best = max(best, int((labels[tuple(replicas.T)] == value).sum()))
        if best == 8:
            break
    return best


def _periodic_labels(pes: np.ndarray, threshold: float) -> np.ndarray:
    """Face-connected components of ``E < t`` in a 2x2x2 tiling, merged across its faces."""
    tiled = np.tile(pes, (2, 2, 2))
    labels, count = ndimage.label(tiled < threshold, structure=FACE_CONNECTIVITY)
    parent = np.arange(count + 1)

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for axis in range(3):
        left = np.take(labels, 0, axis=axis).ravel()
        right = np.take(labels, -1, axis=axis).ravel()
        both = (left > 0) & (right > 0)
        for a, b in set(zip(left[both].tolist(), right[both].tolist())):
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[rb] = ra
    roots = np.array([root(x) for x in range(count + 1)])
    return roots[labels]


def percolation_barriers(pes: np.ndarray, encut: float = 10.0,
                         tol: float = 0.01) -> dict[str, float]:
    """``E_1D``, ``E_2D``, ``E_3D``: lowest thresholds giving 1/2/3D percolation."""
    pes = pes - np.nanmin(pes[np.isfinite(pes)])
    out = {}
    for name, needed in (("E_1D", 2), ("E_2D", 4), ("E_3D", 8)):
        low, high = 0.0, encut
        if percolation_dimension(_periodic_labels(pes, high), pes.shape) < needed:
            out[name] = math.inf
            continue
        while high - low > tol:
            middle = 0.5 * (low + high)
            if percolation_dimension(_periodic_labels(pes, middle), pes.shape) >= needed:
                high = middle
            else:
                low = middle
        out[name] = round(high, 4)
    return out


@dataclass
class EmptyFrameworkResult:
    E_1D: float
    E_2D: float
    E_3D: float
    n_evaluations: int
    grid: tuple[int, int, int]
    seconds: float
    pes: np.ndarray | None = None


def empty_framework_scan(structure, energy_model, mobile: str = "Li", step: float = 0.25,
                         r_min: float = 1.2, encut: float = 10.0,
                         keep_pes: bool = False) -> EmptyFrameworkResult:
    """The ICML pipeline's empty-framework PES and its percolation barriers."""
    from pymatgen.io.ase import AseAtomsAdaptor

    started = time.time()
    matrix = structure.lattice.matrix
    grid = Grid.for_lattice(matrix, step)
    host = structure.copy()
    mobile_sites = [i for i, site in enumerate(host) if site.specie.symbol == mobile]
    probe_number = host[mobile_sites[0]].specie.Z
    host.remove_sites(mobile_sites)
    allowed = allowed_points(matrix, host.frac_coords, grid, r_min)
    energies = np.full(grid.size, np.inf)
    if allowed.any():
        energies[allowed] = energy_model(AseAtomsAdaptor.get_atoms(host), probe_number,
                                         (grid.frac_coords() @ matrix)[allowed])
    pes = energies.reshape(grid.shape)
    barriers = percolation_barriers(pes, encut) if allowed.any() else \
        {"E_1D": math.inf, "E_2D": math.inf, "E_3D": math.inf}
    return EmptyFrameworkResult(**barriers, n_evaluations=int(allowed.sum()), grid=grid.shape,
                                seconds=time.time() - started,
                                pes=pes if keep_pes else None)
