"""
Metrics defined in the CDVAE paper:
Xie, Tian, et al. "Crystal diffusion variational autoencoder for periodic material generation."
arXiv preprint arXiv:2110.06197 (2021).
Code from:
https://github.com/txie-93/cdvae
https://github.com/jiaor17/DiffCSP/
"""
from itertools import product
from collections import Counter
from math import gcd
from typing import Dict, Optional
import logging

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from wrapt_timeout_decorator import timeout
import smact.screening
from pandas import Series
import torch
from pymatgen.core import Composition, Structure
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty


# ---------------------------------------------------------------------------
# Minimal StandardScaler inlined from cdvae_evals so this module has no
# dependency on torch_scatter / torch_sparse.
# ---------------------------------------------------------------------------

class _StandardScaler:
    """Normalizes features of a dataset (numpy version, no torch_scatter dependency)."""

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def transform(self, X):
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        return np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)


# Composition scaler constants (133 elements, from CDVAE/DiffCSP Magpie featurization).
_CompScalerMeans = [
    21.194441759304013, 58.20212663122281, 37.0076848719188,
    36.52738520455582, 13.350626389725019, 29.468922184630255,
    28.71735137747704, 78.8868535524408, 50.16950217496375,
    59.56764743604155, 19.020429484306277, 61.335572740454325,
    47.14515893344343, 141.75135923307818, 94.60620029962553,
    85.95794070476977, 34.07300576173523, 68.06189371516912,
    637.9862061297893, 1817.2394155466848, 1179.2532094169414,
    1127.2743149568837, 431.51034284549826, 909.1060025135899,
    3.7744320927984534, 13.673707104881585, 9.899275012083132,
    9.620186927095652, 3.8426065581251856, 9.96950217496375,
    3.305461575640406, 5.483035282745288, 2.1775737071048815,
    4.215114560306594, 0.8206087101824266, 3.732092798453359,
    109.16732721121315, 179.5570323827936, 70.38970517158047,
    136.0978305229613, 27.027545809538527, 119.16713388110198,
    1.2721433060967857, 2.4614001837260617, 1.1892568776289631,
    1.9844483610247092, 0.4691462290494881, 2.100143582306204,
    1.4829869502174964, 1.9899951667472209, 0.5070082165297245,
    1.7956250375970633, 0.2056251946617602, 1.745867568873852,
    0.05650072498791687, 2.3618656355727405, 2.3053649105848235,
    1.2829636137262992, 0.9995555685850794, 1.5150314161430642,
    0.7731271145480909, 7.4648139197680035, 6.691686805219913,
    4.010677272036105, 2.612307566507693, 3.303528274528758,
    0.2739487675205413, 5.889753504108265, 5.615804736587724,
    2.3244356612494683, 2.1426251769710905, 1.4464475592073465,
    4.739246012566457, 14.578395360077332, 9.839149347510874,
    9.413701584608935, 3.537059747455868, 8.550410826486225,
    0.008119864668922184, 0.43286611889801835, 0.4247462542290962,
    0.16687837041055423, 0.17139889490813626, 0.10898985016916385,
    0.06283228612856452, 2.6573707104881583, 2.594538424359594,
    1.219602938224228, 1.0596390454742999, 1.1120831319478008,
    0.14842919284678588, 3.8473658772353794, 3.6989366843885936,
    1.4541605082183982, 1.3862277372859781, 0.8018849685838569,
    0.03542774287095215, 2.4474625422909617, 2.4120347994200095,
    0.7745217539010397, 0.9145812330586208, 0.3198646689221846,
    1.552730787820203, 6.910681488641856, 5.357950700821653,
    3.615163570754227, 1.9072256165179793, 2.6702271628806185,
    14.608536589568727, 34.83222477045747, 20.223688180890715,
    22.47901710732293, 7.17674504190757, 18.641837024143584,
    0.009066988883518605, 0.9185191396809959, 0.9094521507974755,
    0.4368550481994018, 0.38905942883427047, 0.48375558240695804,
    0.0012985909686158003, 0.21708593995837092, 0.21578734898975546,
    0.08167977375391729, 0.08155386250705281, 0.06036340747305611,
    116.32010633156113, 217.5905751570807, 101.27046882551957,
    162.87154200548844, 41.920624308665566, 136.4664572257129,
]

_CompScalerStds = [
    16.35781741152948, 20.189540126474725, 20.516298414514758,
    16.816765336550194, 7.966591328222124, 22.270791076753067,
    21.802116630115243, 12.804546460581966, 24.756629388687983,
    13.930306216047477, 10.214535652334533, 27.801612936980938,
    39.74031558353379, 54.269739685575814, 53.70466607591569,
    42.852342044453444, 20.78341194242935, 56.28783510219931,
    563.8004405882157, 732.0722574247563, 736.2122907972664,
    606.351603075103, 272.62646060896407, 810.6156779688841,
    3.0362262146833428, 3.2075174256751606, 4.0633818989245665,
    2.9738244769894764, 1.7805586029644034, 5.643243225066782,
    1.1994336274579853, 0.8939013979423364, 1.2297581799896975,
    1.0066021334519983, 0.49129747526397105, 1.4159553146070951,
    31.754756468836774, 28.054241463256226, 38.16336054795611,
    25.83485338379922, 15.388376641904662, 39.67137484594156,
    0.31988340032011076, 0.6833658037760536, 0.7464197945553585,
    0.4881349085029781, 0.3176591553643101, 0.8601748146737138,
    0.5864801661863596, 0.10048913710210677, 0.5836289120986499,
    0.2811748167435902, 0.2468696279341553, 0.5007375747433073,
    0.37237566669029587, 1.7235989187720187, 1.7058836077743305,
    1.1558859351244697, 0.7677842566598179, 1.9203550253462733,
    2.1289400248865182, 3.5326064169848332, 3.708508303762512,
    2.8709941136664567, 1.6110681295257014, 4.310192504023775,
    1.6644182118209292, 6.228287671164213, 6.1200848808512305,
    3.1986202996110302, 2.4492978142248867, 4.030497343977163,
    3.662028270049814, 6.8192125550358345, 6.614243783887738,
    4.334987449618594, 2.568319610320196, 5.9494890200106925,
    0.08974370432893491, 0.4954725441517777, 0.494304434278516,
    0.2309340434963803, 0.2072873961103969, 0.31162647950590266,
    0.39805702757060923, 1.8111691089355726, 1.7973395144505941,
    0.9486995373104102, 0.7538753151875139, 1.5233177017753785,
    0.7952606701778913, 3.711190225170556, 3.638721437232604,
    1.7171165424006831, 1.4307904413917036, 2.1047820817622904,
    0.49193748323158065, 4.064840532426175, 4.035286619587313,
    1.4858577214526643, 1.5799117659864677, 1.6130080156145745,
    1.555249156140194, 4.776932951077492, 4.569790780459629,
    2.224617778217326, 1.7217507416156546, 2.5969733650703763,
    7.215001918238936, 19.252513469778584, 18.775394044177858,
    9.447222764774764, 6.7467931836261235, 11.106825644766616,
    0.27206794253092115, 1.6449321034573106, 1.6236282792648686,
    0.8506917026741503, 0.7020945355184042, 1.2281895279350408,
    0.04134438177238229, 0.5508855867341717, 0.5486095551438679,
    0.24239297524046477, 0.2127779137935831, 0.3036750942874694,
    80.06063945615361, 21.345794811194104, 80.16475677581042,
    52.58533928558554, 35.40836791039412, 85.980205895116,
]

CompScaler = _StandardScaler(
    means=np.array(_CompScalerMeans),
    stds=np.array(_CompScalerStds),
    replace_nan_token=0.)

logger = logging.getLogger(__name__)


COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie', impute_nan=False)

class Crystal(object):
    @classmethod
    def from_pymatgen(cls, structure: Structure):
        crys_array_dict = {
            'frac_coords': structure.frac_coords,
            'atom_types': np.array([s.number for s in structure.species]),
            'lengths': np.array(structure.lattice.abc),
            'angles': np.array(structure.lattice.angles)
        }
        return cls(crys_array_dict, structure)

    def __init__(self,
        crys_array_dict: Dict[str, np.ndarray],
        structure: Optional[Structure] = None):

        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)

        if structure is not None:
            self.structure = structure
            self.constructed = True
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()


    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = timed_smact_validity_from_composition(self.structure.composition)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            logger.debug("Fingerprint construction failed.")
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)



def smact_validity_optimised(
                   elem_symbols, count,
                   use_pauling_test=True,
                   include_alloys=True,
                   apply_gcd=False):
    element_set = frozenset(elem_symbols)
    if len(element_set) == 1:
        return True
    if include_alloys and element_set.issubset(smact.metals):
        return True

    try:
        space = smact.element_dictionary(elem_symbols)
    except NameError:
        # NameError: Elemental data for Hs not found.
        return False
    electronegs = [e.pauling_eneg for e in space.values()]
    ox_combos = [e.oxidation_states for e in space.values()]
    if any(ox is None for ox in ox_combos):
        return False

    if apply_gcd:
        gcd_count = gcd(*count)
        count = tuple(c // gcd_count for c in count)
    threshold = max(count)
    stoichs = [(c,) for c in count]
    for ox_states in product(*ox_combos):
        cn_r = smact.neutral_ratios_iter(ox_states, stoichs=stoichs, threshold=threshold)
        if any(True for _ in cn_r):
            if not use_pauling_test:
                return True
            try:
                if smact.screening.pauling_test(ox_states, electronegs, symbols=elem_symbols):
                    return True
            except TypeError:
                # if no electronegativity data, assume it is okay
                return True
    return False


# Pandas tries to treat the decorated function as iterable, which causes a TypeError.
@timeout(15)
def timed_smact_validity_optimised(*args, **kwargs) -> bool:
    """
    Computes the SMACT validity of a record in pyxtal.from_random arguments format.
    If the computation takes longer than 15 seconds, returns False.
    """
    try:
        return smact_validity_optimised(*args, **kwargs)
    except TimeoutError:
        logger.debug("SMAC-T validity timed out, returning False")
        return False


def timed_smact_validity_from_composition(composition: Composition, apply_gcd: bool=True) -> bool:
    """
    Computes the SMACT validity of a record in pyxtal.from_random arguments format.
    If the computation takes longer than 15 seconds, returns False.
    """
    get_el_amt_dict = composition.get_el_amt_dict()
    return timed_smact_validity_optimised(
        elem_symbols=list(get_el_amt_dict.keys()),
        count=list(map(int, get_el_amt_dict.values())),
        apply_gcd=apply_gcd)


def timed_smact_validity_from_record(record: Dict|Series, apply_gcd: bool=True) -> bool:
    """
    Computes the SMACT validity of a record in pyxtal.from_random arguments format.
    If the computation takes longer than 15 seconds, returns False.
    """
    return timed_smact_validity_optimised(record["species"], record["numIons"], apply_gcd=apply_gcd)


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.full(dist_mat.shape[0], cutoff + 10.))
    if crystal.volume < 0.1 or dist_mat.min() < cutoff:
        return False
    return True

def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise ValueError(
                f'Not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}


    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}


    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            try:
                from cdvae_property_models import prop_model_eval
            except ImportError:
                raise ImportError(
                    "cdvae-property-models is required for property evaluation. "
                    "Install it with: pip install cdvae-property-models"
                ) from None
            with torch.no_grad():
                pred_props = prop_model_eval(self.eval_model_name, [
                                            c.dict for c in self.valid_samples])
                gt_props = prop_model_eval(self.eval_model_name, [
                                        c.dict for c in self.gt_crys])
                wdist_prop = wasserstein_distance(pred_props, gt_props)
                return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        cov_metrics_dict = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict


def compute_cov(crys, gt_crys,
                struc_cutoff, comp_cutoff, num_gen_crystals=None):
    struc_fps = [c.struct_fp for c in crys]
    comp_fps = [c.comp_fp for c in crys]
    gt_struc_fps = [c.struct_fp for c in gt_crys]
    gt_comp_fps = [c.comp_fp for c in gt_crys]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)

    comp_fps = CompScaler.transform(comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
    }

    return metrics_dict
