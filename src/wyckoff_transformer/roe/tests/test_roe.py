"""Tests for the rules of engagement: composition, bookkeeping, and accounting.

The filters and the reconstruction are covered where they live.  What is tested
here is what this package adds: that a cohort cannot lose a gene or resurrect
one, that a mode refuses an assembly that cannot run before anything expensive
does, and that the per-sampled-gene restatement uses the denominator it claims.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from wyckoff_transformer.roe.cohort import (
    COHORT_FILE,
    ENGAGED_GENES_FILE,
    MANIFEST_FILE,
    Cohort,
    ResurrectionError,
)
from wyckoff_transformer.roe.components import check_requirements
from wyckoff_transformer.roe.plan import (
    BROADSIDE,
    FIRE_CONTROL,
    FIRE_DISCIPLINE,
    RULES_OF_ENGAGEMENT,
    TORPEDO_RUN,
    Engagement,
    RulesOfEngagement,
    resolve,
)
from wyckoff_transformer.roe.report import (
    cohort_screen,
    engagement_report,
    reconstruction_cost,
)

NACL = {"group": 225, "species": ["Na", "Cl"], "numIons": [4, 4], "sites": [["4a"], ["4b"]]}
NACL_REORDERED = {
    "group": 225, "species": ["Cl", "Na"], "numIons": [4, 4], "sites": [["4b"], ["4a"]],
}
OTHER = {"group": 221, "species": ["Sr", "Ti", "O"], "numIons": [1, 1, 3],
         "sites": [["1a"], ["1b"], ["3c"]]}
ILLEGAL = {"group": 225, "species": ["Na"], "numIons": [16], "sites": [["16z"]]}


# --------------------------------------------------------------------------- #
# Fakes.  A mode is a composition, so the composition has to be testable without
# a checkpoint, a reference archive or a GPU.
# --------------------------------------------------------------------------- #
class FakeSource:
    name = "fake_source"
    requires_plan = False

    def __init__(self, genes):
        self.genes = list(genes)
        self.asked = None

    def attempts(self, n):
        return n

    def draw(self, n_structures, plan=None):
        self.asked = (n_structures, plan)
        return self.genes[:n_structures]

    def describe(self):
        return {"component": self.name}


class FakeFilter:
    def __init__(self, name, slot, keep, provides=(), requires=()):
        self.name = name
        self.slot = slot
        self.keep = keep
        self.provides = tuple(provides)
        self.requires = tuple(requires)

    def apply(self, cohort):
        surviving = [i for i in cohort.kept_indices() if i in self.keep]
        cohort.record(
            "filter", self.name, passed=surviving,
            columns={name: pd.Series(1.0, index=cohort.table.index) for name in self.provides})

    def describe(self):
        return {"component": self.name}


class FakeSampler:
    name = "fake_sampler"

    def __init__(self):
        self.asked = None

    def plan(self, n):
        self.asked = n

        class _Plan(list):
            element_symbols = ()
            query: dict = {}

        return _Plan(range(n))

    def describe(self):
        return {"component": self.name}


class RecordingReconstructor:
    name = "fake_yard"

    def __init__(self):
        self.seen = None

    def reconstruct(self, cohort, output_dir):
        self.seen = cohort.kept_indices()
        return {"free": {"metasun_per_sampled_gene": 0.5}}

    def describe(self):
        return {"component": self.name}


# --------------------------------------------------------------------------- #
# Cohort bookkeeping
# --------------------------------------------------------------------------- #
def test_cohort_marks_and_never_drops():
    cohort = Cohort.from_genes([NACL, OTHER, ILLEGAL])
    cohort.record("filter", "first", passed=[0, 1])
    assert cohort.n_sampled == 3, "a filter must not change the denominator"
    assert cohort.kept_indices() == [0, 1]
    assert len(cohort.table) == 3
    assert bool(cohort.table.loc[2, "passed_first"]) is False


def test_a_filter_cannot_resurrect_a_dropped_gene():
    cohort = Cohort.from_genes([NACL, OTHER, ILLEGAL])
    cohort.record("filter", "first", passed=[0])
    with pytest.raises(ResurrectionError):
        cohort.record("filter", "second", passed=[0, 1])


def test_a_row_a_filter_never_saw_gets_no_verdict():
    cohort = Cohort.from_genes([NACL, OTHER, ILLEGAL])
    cohort.record("filter", "first", passed=[0, 1])
    cohort.record("filter", "second", passed=[0])
    assert pd.isna(cohort.table.loc[2, "passed_second"]), (
        "the second filter never looked at gene 2; recording False would read as a verdict")


def test_weight_survives_deduplication():
    cohort = Cohort.from_genes([NACL, NACL_REORDERED, OTHER])
    cohort.record("filter", "screen", passed=[0, 2], weights={0: 2, 1: 0, 2: 1})
    assert cohort.weight_kept == 3, "every sampled gene must still be behind some kept row"
    assert int(cohort.table["weight"].sum()) == 3


def test_writing_a_cohort_produces_the_three_files(tmp_path):
    cohort = Cohort.from_genes([NACL, OTHER, ILLEGAL])
    cohort.record("filter", "screen", passed=[0, 2])
    written = cohort.write(tmp_path)
    assert set(p.name for p in written.values()) == {
        ENGAGED_GENES_FILE, COHORT_FILE, MANIFEST_FILE}
    assert cohort.table.loc[0, "engaged_index"] == 0
    assert cohort.table.loc[2, "engaged_index"] == 1
    assert pd.isna(cohort.table.loc[1, "engaged_index"])
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text())
    assert manifest["sampled"] == 3 and manifest["kept"] == 2


# --------------------------------------------------------------------------- #
# The modes themselves
# --------------------------------------------------------------------------- #
def test_the_four_modes_escalate():
    """Each mode runs the filters of the one before it, plus at most one more."""
    assert BROADSIDE.filters == ()
    assert FIRE_DISCIPLINE.filters == ("screen",)
    assert FIRE_CONTROL.filters == ("screen", "energy")
    assert set(TORPEDO_RUN.filters) == {"screen", "energy"}
    assert TORPEDO_RUN.filters == ("energy", "screen"), (
        "a torpedo run ranges before it screens; that ordering is the mode")
    assert TORPEDO_RUN.needs_sampler
    assert not any(roe.needs_sampler for roe in
                   (BROADSIDE, FIRE_DISCIPLINE, FIRE_CONTROL))


def test_only_broadside_is_charged_for_duplicates():
    charged = {roe.name for roe in RULES_OF_ENGAGEMENT.values() if roe.charge_duplicates}
    assert charged == {"broadside"}, (
        "a mode with a uniqueness screen pays for the genes it kept, not the draws")


def test_an_unknown_mode_names_the_known_ones():
    with pytest.raises(ValueError, match="fire-discipline"):
        resolve("enfilade")


def test_a_slot_cannot_be_named_twice():
    with pytest.raises(ValueError, match="more than once"):
        RulesOfEngagement("x", "", ("screen", "screen"), False, False)


def test_an_unknown_slot_is_refused():
    with pytest.raises(ValueError, match="unknown filter slot"):
        RulesOfEngagement("x", "", ("ballast",), False, False)


# --------------------------------------------------------------------------- #
# Assembly is validated before anything expensive runs
# --------------------------------------------------------------------------- #
def test_a_missing_slot_is_refused():
    with pytest.raises(ValueError, match="nothing was given"):
        Engagement(roe=FIRE_CONTROL, source=FakeSource([NACL]),
                   filters={"screen": FakeFilter("s", "screen", {0})})


def test_a_filter_with_no_slot_in_the_mode_is_refused():
    with pytest.raises(ValueError, match="has no slot"):
        Engagement(roe=BROADSIDE, source=FakeSource([NACL]),
                   filters={"screen": FakeFilter("s", "screen", {0})})


def test_a_filter_in_the_wrong_slot_is_refused():
    with pytest.raises(ValueError, match="was given the"):
        Engagement(roe=FIRE_DISCIPLINE, source=FakeSource([NACL]),
                   filters={"screen": FakeFilter("e", "energy", {0})})


def test_a_torpedo_run_needs_a_sampler():
    with pytest.raises(ValueError, match="chemical-system sampler"):
        Engagement(
            roe=TORPEDO_RUN, source=FakeSource([NACL]),
            filters={"screen": FakeFilter("s", "screen", {0}),
                     "energy": FakeFilter("e", "energy", {0})})


def test_a_sampler_in_a_mode_that_does_not_aim_is_refused():
    with pytest.raises(ValueError, match="torpedo-run"):
        Engagement(roe=BROADSIDE, source=FakeSource([NACL]), sampler=FakeSampler())


def test_a_filter_chain_missing_its_input_is_refused():
    with pytest.raises(ValueError, match="needs cohort column"):
        check_requirements([
            FakeFilter("ranker", "energy", {0}, requires=("surprisal",)),
        ])


def test_a_filter_chain_whose_inputs_are_produced_upstream_is_accepted():
    check_requirements([
        FakeFilter("scorer", "surprisal", {0}, provides=("surprisal",)),
        FakeFilter("ranker", "energy", {0}, requires=("surprisal",)),
    ])


# --------------------------------------------------------------------------- #
# End to end, with fakes
# --------------------------------------------------------------------------- #
def test_a_mode_runs_its_filters_in_the_order_it_names(tmp_path):
    order = []

    class Noting(FakeFilter):
        def apply(self, cohort):
            order.append(self.slot)
            super().apply(cohort)

    engagement = Engagement(
        roe=TORPEDO_RUN,
        source=FakeSource([NACL, OTHER, NACL_REORDERED]),
        filters={"screen": Noting("s", "screen", {0, 1, 2}),
                 "energy": Noting("e", "energy", {0, 1, 2})},
        sampler=FakeSampler(),
        reconstructor=RecordingReconstructor(),
    )
    engagement.run(3, tmp_path)
    assert order == ["energy", "screen"]


def test_the_reconstructor_only_sees_what_survived(tmp_path):
    yard = RecordingReconstructor()
    engagement = Engagement(
        roe=FIRE_DISCIPLINE,
        source=FakeSource([NACL, OTHER, NACL_REORDERED]),
        filters={"screen": FakeFilter("s", "screen", {0, 2})},
        reconstructor=yard,
    )
    cohort = engagement.run(3, tmp_path)
    assert yard.seen == [0, 2]
    assert cohort.n_sampled == 3


def test_the_sampler_plans_one_request_per_draw_not_per_gene(tmp_path):
    class Oversampling(FakeSource):
        def attempts(self, n):
            return n + 5

    sampler = FakeSampler()
    engagement = Engagement(
        roe=TORPEDO_RUN,
        source=Oversampling([NACL] * 20),
        filters={"screen": FakeFilter("s", "screen", set(range(20))),
                 "energy": FakeFilter("e", "energy", set(range(20)))},
        sampler=sampler,
        reconstructor=None,
    )
    engagement.run(10, tmp_path)
    assert sampler.asked == 15


# --------------------------------------------------------------------------- #
# Accounting
# --------------------------------------------------------------------------- #
def _written_cohort(tmp_path, weights=None):
    cohort = Cohort.from_genes([NACL, NACL_REORDERED, OTHER, ILLEGAL])
    cohort.record("filter", "screen", passed=[0, 2], weights=weights or {0: 2, 1: 0, 2: 1, 3: 1})
    cohort.write(tmp_path)
    return cohort


def test_the_engaged_screen_carries_the_cohorts_denominator(tmp_path):
    cohort = _written_cohort(tmp_path)
    screen = cohort_screen(cohort)
    assert screen.n_sampled == 4, "rates must be per gene drawn, not per gene reconstructed"
    assert screen.counts == {0: 2, 1: 1}, (
        "keyed by position in the engaged file, valued by the draws behind it")


def test_broadside_is_charged_for_the_duplicates_it_never_noticed(tmp_path):
    cohort = _written_cohort(tmp_path)
    relaxations = pd.DataFrame(
        {"index": [0, 0, 1], "trial": [0, 1, 0]}).set_index("index")
    charged = reconstruction_cost(cohort, relaxations, charge_duplicates=True)
    assert charged["trials_run"] == 3
    # Engaged row 0 stands for two draws, and each of its two trials is charged twice.
    assert charged["trials_charged"] == 5
    kept = reconstruction_cost(cohort, relaxations, charge_duplicates=False)
    assert kept["trials_charged"] == 3


def test_the_report_restates_the_funnel_against_the_drawn_cohort(tmp_path):
    cohort = _written_cohort(tmp_path)
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    # Engaged row 0 is a MetaSUN and stands for 2 of the 4 drawn genes.
    pd.DataFrame({
        "index": [0, 1],
        "has_structure": [True, True],
        "valid_structure": [True, True],
        "unique_structure": [True, True],
        "novel_structure": [True, False],
        "e_above_hull": [0.05, 0.4],
    }).set_index("index").to_csv(protocol_dir / "structures.csv")
    pd.DataFrame({"index": [0, 0, 1], "trial": [0, 1, 0]}).set_index("index").to_csv(
        protocol_dir / "relaxations.csv")

    report = engagement_report(cohort, FIRE_DISCIPLINE, tmp_path)
    free = report["per_sampled_gene"]["free"]
    assert free["metastable_among_novel"] == 1
    assert free["metasun_per_sampled_gene"] == pytest.approx(2 / 4), (
        "one hit standing for two draws, over the four genes the mode drew")
    assert report["trials_per_hit"]["free"]["metasun"] == pytest.approx(3 / 1)
    assert report["cohort"]["sampled"] == 4
    assert report["cohort"]["engaged"] == 2


# --------------------------------------------------------------------------- #
# CLI plumbing
# --------------------------------------------------------------------------- #
def test_protocol_arguments_are_split_off_at_the_bare_separator():
    from wyckoff_transformer.roe.cli import split_protocol_argv

    ours, theirs = split_protocol_argv(
        ["run", "fire-control", "--output-dir", "x", "--", "--devices", "cuda:0"])
    assert ours == ["run", "fire-control", "--output-dir", "x"]
    assert theirs == ["--devices", "cuda:0"]
    assert split_protocol_argv(["list"]) == (["list"], [])


# --------------------------------------------------------------------------- #
# The real always-applied step and the real screen, on three hand-checked genes
# --------------------------------------------------------------------------- #
def test_validity_drops_only_the_illegal_gene():
    from wyckoff_transformer.roe.builtin import apply_validity

    cohort = Cohort.from_genes([NACL, OTHER, ILLEGAL])
    apply_validity(cohort)
    assert cohort.kept_indices() == [0, 1]
    assert "16z" in str(cohort.table.loc[2, "invalid_reason"]) or pd.notna(
        cohort.table.loc[2, "invalid_reason"])
    assert cohort.history[-1].stage == "validity"


def test_the_screen_removes_a_duplicate_and_a_known_gene():
    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter
    from wyckoff_transformer.roe.builtin import NoveltyUniquenessScreen

    fingerprinter = GeneFingerprinter()
    screen = NoveltyUniquenessScreen()
    # Stand in for the archive: it holds the perovskite and not the rock salt.
    screen._reference = {fingerprinter.fingerprint(OTHER)}

    cohort = Cohort.from_genes([NACL, NACL_REORDERED, OTHER])
    screen.apply(cohort)

    assert cohort.kept_indices() == [0], (
        "the reordered gene is the same gene, and the perovskite is already known")
    assert cohort.table.loc[0, "duplicates"] == 2
    assert bool(cohort.table.loc[0, "gene_novel"]) is True
    assert bool(cohort.table.loc[1, "gene_novel"]) is True, (
        "novelty is a property of the fingerprint, so a duplicate carries it too")
    assert bool(cohort.table.loc[2, "gene_novel"]) is False
    assert cohort.table.loc[0, "weight"] == 2, "the representative stands for both draws"
    assert int(cohort.table["weight"].sum()) == 3, "no draw is lost from the denominator"
    assert cohort.weight_kept == 2, (
        "two of the three draws are behind the kept row; the known one is not")


def test_the_screen_can_check_uniqueness_without_a_reference():
    from wyckoff_transformer.roe.builtin import NoveltyUniquenessScreen

    screen = NoveltyUniquenessScreen(novelty=False)
    cohort = Cohort.from_genes([NACL, NACL_REORDERED, OTHER])
    screen.apply(cohort)
    assert cohort.kept_indices() == [0, 2], "no archive was consulted, so nothing is known"
    assert screen.reference() == set()


def test_a_screen_that_checks_nothing_is_refused():
    from wyckoff_transformer.roe.builtin import NoveltyUniquenessScreen

    with pytest.raises(ValueError, match="does nothing"):
        NoveltyUniquenessScreen(uniqueness=False, novelty=False)


def test_the_screen_deduplicates_among_survivors_not_among_all_genes():
    """A class must not be lost because the member that came first was filtered out."""
    from wyckoff_transformer.roe.builtin import NoveltyUniquenessScreen

    screen = NoveltyUniquenessScreen(novelty=False)
    cohort = Cohort.from_genes([NACL, NACL_REORDERED, OTHER])
    cohort.record("filter", "energy", passed=[1, 2])
    screen.apply(cohort)
    assert cohort.kept_indices() == [1, 2], (
        "gene 0 was already gone, so gene 1 is the representative of its class")


# --------------------------------------------------------------------------- #
# The two screen backends must be interchangeable
# --------------------------------------------------------------------------- #
def _screen_pair(known_genes):
    """A python-backed and a tensor-backed screen over the same tiny archive."""
    from wyckoff_transformer.evaluation.gene_hash import GeneKeyTable, gene_keys
    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter
    from wyckoff_transformer.roe.builtin import NoveltyUniquenessScreen

    fingerprinter = GeneFingerprinter()
    records = [fingerprinter.record(gene) for gene in known_genes]

    python_screen = NoveltyUniquenessScreen(backend="python")
    python_screen._reference = {
        __import__("wyckoff_transformer.evaluation.novelty", fromlist=["x"])
        .record_to_augmented_fingerprint(record) for record in records}

    tensor_screen = NoveltyUniquenessScreen(backend="tensor")
    tensor_screen._table = GeneKeyTable.from_keys(gene_keys(records))
    return python_screen, tensor_screen


@pytest.mark.parametrize("known", [[], [OTHER], [NACL], [NACL, OTHER]])
def test_the_two_screen_backends_agree(known):
    cohort_genes = [NACL, NACL_REORDERED, OTHER, NACL]
    python_screen, tensor_screen = _screen_pair(known)

    verdicts = []
    for screen in (python_screen, tensor_screen):
        cohort = Cohort.from_genes(cohort_genes)
        screen.apply(cohort)
        verdicts.append((
            cohort.kept_indices(),
            cohort.table["duplicates"].fillna(-1).tolist(),
            cohort.table["gene_novel"].fillna(False).tolist(),
            cohort.table["weight"].tolist(),
        ))
    assert verdicts[0] == verdicts[1], (
        f"the backends disagree with {len(known)} gene(s) in the archive")


def test_the_tensor_backend_reports_itself_in_the_manifest():
    _, tensor_screen = _screen_pair([OTHER])
    cohort = Cohort.from_genes([NACL, OTHER])
    tensor_screen.apply(cohort)
    assert cohort.history[-1].detail["backend"] == "tensor"
    assert tensor_screen.describe()["backend"] == "tensor"


def test_an_unknown_screen_backend_is_refused():
    from wyckoff_transformer.roe.builtin import NoveltyUniquenessScreen

    with pytest.raises(ValueError, match="backend is one of"):
        NoveltyUniquenessScreen(backend="cuda")


def test_the_tensor_backend_drops_an_illegal_gene_like_the_python_one():
    python_screen, tensor_screen = _screen_pair([])
    for screen in (python_screen, tensor_screen):
        cohort = Cohort.from_genes([NACL, ILLEGAL, OTHER])
        screen.apply(cohort)
        assert cohort.kept_indices() == [0, 2]
        assert cohort.history[-1].detail["invalid"] == 1


# --------------------------------------------------------------------------- #
# A fixed reconstruction budget
# --------------------------------------------------------------------------- #
class StreamingSource:
    """Draws fresh genes every call, so a top-up round can make progress."""

    name = "streaming"
    requires_plan = False

    def __init__(self, keep_every=2, exhaust_after=None):
        self.drawn = 0
        self.keep_every = keep_every
        self.exhaust_after = exhaust_after
        self.rounds = 0

    def attempts(self, n):
        return n

    def draw(self, n_structures, plan=None):
        self.rounds += 1
        if self.exhaust_after is not None and self.rounds > self.exhaust_after:
            return []
        genes = []
        for _ in range(n_structures):
            genes.append({"group": 225, "species": ["Na"], "numIons": [4],
                          "sites": [["4a"]], "serial": self.drawn})
            self.drawn += 1
        return genes

    def describe(self):
        return {"component": self.name}


class EveryNth:
    """Keeps genes whose serial is divisible by *n*: a fixed, known pass rate."""

    name = "every_nth"
    slot = "screen"
    provides = ()
    requires = ()

    def __init__(self, n):
        self.n = n

    def apply(self, cohort):
        passed = [i for i in cohort.kept_indices()
                  if cohort.genes[i]["serial"] % self.n == 0]
        cohort.record("filter", self.name, passed=passed)

    def describe(self):
        return {"component": self.name, "n": self.n}


def test_a_budget_is_reached_by_drawing_more(tmp_path):
    source = StreamingSource()
    engagement = Engagement(
        roe=FIRE_DISCIPLINE, source=source, filters={"screen": EveryNth(4)})
    cohort = engagement.run(20, tmp_path, target_engaged=25)
    assert cohort.n_kept == 25, "the budget is what makes two modes comparable"
    assert source.rounds > 1, "one batch of 20 cannot yield 25 survivors at a 1-in-4 rate"
    assert cohort.n_sampled == 97, "serial 96 is the 25th multiple of four"


def test_a_surplus_is_truncated_to_the_budget(tmp_path):
    engagement = Engagement(
        roe=FIRE_DISCIPLINE, source=StreamingSource(), filters={"screen": EveryNth(2)})
    cohort = engagement.run(100, tmp_path, target_engaged=10)
    assert cohort.n_kept == 10
    assert cohort.n_sampled == 19, "the tenth even serial is 18, so 19 draws were needed"


def test_the_budget_keeps_a_prefix_of_the_cohort(tmp_path):
    engagement = Engagement(
        roe=FIRE_DISCIPLINE, source=StreamingSource(), filters={"screen": EveryNth(2)})
    cohort = engagement.run(20, tmp_path, target_engaged=3)
    assert cohort.kept_indices() == [0, 2, 4], (
        "truncation must keep the earliest survivors, so the cohort stays a sample")


def test_an_exhausted_source_stops_the_loop(tmp_path):
    engagement = Engagement(
        roe=FIRE_DISCIPLINE, source=StreamingSource(exhaust_after=1),
        filters={"screen": EveryNth(10)})
    cohort = engagement.run(10, tmp_path, target_engaged=1000)
    assert cohort.n_kept == 1, "serial 0 is the only one of ten that passes"
    assert cohort.n_sampled == 10, "the second round returned nothing to add"


def test_the_round_cap_stops_the_loop(tmp_path):
    class KeepsNothing(EveryNth):
        def apply(self, cohort):
            cohort.record("filter", self.name, passed=[])

    engagement = Engagement(
        roe=FIRE_DISCIPLINE, source=StreamingSource(), filters={"screen": KeepsNothing(1)})
    cohort = engagement.run(10, tmp_path, target_engaged=5, max_rounds=3)
    assert cohort.n_kept == 0
    assert cohort.n_sampled == 30, "nothing passed, so each top-up is another first batch"


def test_the_top_up_batch_is_capped(tmp_path):
    """One survivor must not make the next batch a thousand times the first."""
    class OnlyTheFirst(EveryNth):
        def apply(self, cohort):
            kept = cohort.kept_indices()
            cohort.record("filter", self.name, passed=kept[:1])

    source = StreamingSource()
    engagement = Engagement(
        roe=FIRE_DISCIPLINE, source=source, filters={"screen": OnlyTheFirst(1)})
    engagement.run(10, tmp_path, target_engaged=500, max_rounds=2)
    assert source.drawn == 110, "10, then the cap of 10x the first batch"


def test_no_target_draws_once(tmp_path):
    source = StreamingSource()
    engagement = Engagement(
        roe=FIRE_DISCIPLINE, source=source, filters={"screen": EveryNth(2)})
    cohort = engagement.run(10, tmp_path)
    assert source.rounds == 1
    assert cohort.n_sampled == 10


def test_uniqueness_is_judged_across_rounds_not_within_one(tmp_path):
    """A round-three duplicate of a round-one gene must still be caught."""
    from wyckoff_transformer.roe.builtin import NoveltyUniquenessScreen

    class Repeating:
        name = "repeating"
        requires_plan = False

        def __init__(self):
            self.rounds = 0

        def attempts(self, n):
            return n

        def draw(self, n_structures, plan=None):
            self.rounds += 1
            # The same two genes every round: after round one, nothing is new.
            return [NACL, OTHER][:n_structures] * max(1, n_structures // 2)

        def describe(self):
            return {"component": self.name}

    screen = NoveltyUniquenessScreen(novelty=False)
    engagement = Engagement(roe=FIRE_DISCIPLINE, source=Repeating(),
                            filters={"screen": screen})
    cohort = engagement.run(2, tmp_path, target_engaged=10, max_rounds=3)
    assert cohort.n_kept == 2, (
        "there are only two distinct genes however many rounds are drawn")
    assert cohort.n_sampled > 2


# --------------------------------------------------------------------------- #
# A shared pool, handed out in order
# --------------------------------------------------------------------------- #
def _pool(tmp_path, genes):
    import gzip

    path = tmp_path / "pool.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(genes, handle)
    return path


def test_the_pool_is_handed_out_in_successive_slices(tmp_path):
    from wyckoff_transformer.roe.builtin import GeneFileSource

    source = GeneFileSource(_pool(tmp_path, [{"i": i} for i in range(10)]))
    assert source.draw(4) == [{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}]
    assert source.draw(3) == [{"i": 4}, {"i": 5}, {"i": 6}]
    assert source.draw(10) == [{"i": 7}, {"i": 8}, {"i": 9}]
    assert source.draw(1) == [], "an exhausted pool hands out nothing"
    assert source.consumed == 10


def test_a_mode_consumes_only_as_much_of_the_pool_as_its_budget_needs(tmp_path):
    from wyckoff_transformer.roe.builtin import GeneFileSource

    genes = [{"group": 225, "species": ["Na"], "numIons": [4], "sites": [["4a"]],
              "serial": i} for i in range(200)]
    source = GeneFileSource(_pool(tmp_path, genes))
    engagement = Engagement(
        roe=FIRE_DISCIPLINE, source=source, filters={"screen": EveryNth(4)})
    cohort = engagement.run(20, tmp_path / "out", target_engaged=10)
    assert cohort.n_kept == 10
    assert cohort.n_sampled <= source.consumed < 200, (
        "the rest of the pool is left for another arm of the same comparison")


# --------------------------------------------------------------------------- #
# Selecting on the predicted hull
# --------------------------------------------------------------------------- #
class CannedHullFilter:
    """The real filter's selection logic over canned scores."""

    def __new__(cls, scores, **kwargs):
        from wyckoff_transformer.roe.builtin import PredictedHullFilter

        instance = PredictedHullFilter.__new__(PredictedHullFilter)
        PredictedHullFilter.__init__(instance, regressor=None, reference=None, **kwargs)
        instance._score = lambda cohort, indices: pd.DataFrame({
            "formula": ["X"] * len(indices),
            "predicted_formation_energy": [0.0] * len(indices),
            "hull_energy": [0.0] * len(indices),
            "score": [scores[i] for i in indices],
        }, index=pd.Index(indices, name="index"))
        return instance


def _cohort_of(n):
    return Cohort.from_genes([{"i": i} for i in range(n)])


def test_a_threshold_keeps_everything_at_or_below_the_margin():
    scores = {0: -0.2, 1: -0.01, 2: 0.0, 3: 0.05, 4: 0.9}
    cohort = _cohort_of(5)
    CannedHullFilter(scores, select="threshold", margin=0.0).apply(cohort)
    assert cohort.kept_indices() == [0, 1, 2]


def test_top_k_keeps_exactly_the_budget_however_few_clear_the_hull():
    scores = {0: 0.9, 1: -0.2, 2: 0.3, 3: 0.05, 4: 0.4}
    cohort = _cohort_of(5)
    gene_filter = CannedHullFilter(scores, select="top", top_k=3)
    gene_filter.apply(cohort)
    assert cohort.kept_indices() == [1, 2, 3], (
        "the three lowest predicted e_hull, in cohort order")
    detail = cohort.history[-1].detail
    assert detail["below_hull"] == 1, "only one is actually predicted below the hull"
    assert detail["selection_cut_e_hull"] == pytest.approx(0.3), (
        "the manifest must say what the budget's cut actually landed at")


def test_top_k_larger_than_the_pool_keeps_the_pool():
    cohort = _cohort_of(3)
    CannedHullFilter({0: 0.1, 1: 0.2, 2: 0.3}, select="top", top_k=10).apply(cohort)
    assert cohort.kept_indices() == [0, 1, 2]


def test_an_undecided_gene_ranks_last_under_top_k():
    """No hull to compare with is not evidence of a low energy."""
    import numpy as np

    scores = {0: 0.5, 1: np.nan, 2: 0.2}
    cohort = _cohort_of(3)
    CannedHullFilter(scores, select="top", top_k=2).apply(cohort)
    assert cohort.kept_indices() == [0, 2]


def test_an_undecided_gene_is_kept_when_the_budget_is_not_filled_without_it():
    import numpy as np

    cohort = _cohort_of(3)
    CannedHullFilter({0: 0.5, 1: np.nan, 2: 0.2}, select="top", top_k=3).apply(cohort)
    assert cohort.kept_indices() == [0, 1, 2]


def test_an_undecided_gene_can_be_dropped_outright():
    import numpy as np

    cohort = _cohort_of(3)
    CannedHullFilter({0: 0.5, 1: np.nan, 2: 0.2},
                     select="top", top_k=3, on_missing_hull="drop").apply(cohort)
    assert cohort.kept_indices() == [0, 2]


def test_top_selection_needs_a_budget():
    from wyckoff_transformer.roe.builtin import PredictedHullFilter

    with pytest.raises(ValueError, match="needs top_k"):
        PredictedHullFilter(None, None, select="top")


def test_an_unknown_selector_is_refused():
    from wyckoff_transformer.roe.builtin import PredictedHullFilter

    with pytest.raises(ValueError, match="select is one of"):
        PredictedHullFilter(None, None, select="argmin")


def test_a_gene_is_scored_once_however_many_rounds_it_survives(monkeypatch):
    """The top-up loop re-filters the whole cohort, so scoring must be cached."""
    from wyckoff_transformer.roe import builtin

    calls = []

    def fake_score_genes(genes, regressor, reference, augmentation_samples=1,
                         hull_lookup=None):
        calls.append(len(genes))
        return pd.DataFrame({
            "formula": ["X"] * len(genes),
            "predicted_formation_energy": [0.0] * len(genes),
            "hull_energy": [0.0] * len(genes),
            "score": [-1.0] * len(genes),
        }, index=pd.RangeIndex(len(genes)))

    monkeypatch.setattr("wyckoff_transformer.cli.gene_screen.score_genes", fake_score_genes)
    gene_filter = builtin.PredictedHullFilter(None, None)
    gene_filter._lookup = object()

    cohort = Cohort.from_genes([NACL, OTHER, PEROVSKITE_LIKE])
    gene_filter.apply(cohort)
    gene_filter.apply(cohort)
    assert calls == [3], "the second pass must score nothing new"


PEROVSKITE_LIKE = {"group": 221, "species": ["Ba", "Ti", "O"], "numIons": [1, 1, 3],
                   "sites": [["1a"], ["1b"], ["3c"]]}


def test_draws_made_after_the_budget_was_filled_are_not_charged(tmp_path):
    """A mode is charged for the generation it needed, not for the batch size."""
    engagement = Engagement(
        roe=FIRE_DISCIPLINE, source=StreamingSource(), filters={"screen": EveryNth(2)})
    cohort = engagement.run(100, tmp_path, target_engaged=3)
    assert cohort.kept_indices() == [0, 2, 4]
    assert cohort.n_sampled == 5, (
        "the budget was filled by the fifth draw; the other 95 were never needed")


def test_an_exact_fit_needs_no_trimming(tmp_path):
    engagement = Engagement(
        roe=FIRE_DISCIPLINE, source=StreamingSource(), filters={"screen": EveryNth(1)})
    cohort = engagement.run(10, tmp_path, target_engaged=10)
    assert cohort.n_sampled == 10 and cohort.n_kept == 10
    assert [r.component for r in cohort.history][-1] != "budget"


# --------------------------------------------------------------------------- #
# A budget taken by rank, after every filter
# --------------------------------------------------------------------------- #
class RankingFilter:
    """Scores every gene and keeps them all, leaving the cut to the budget."""

    name = "ranker"
    slot = "energy"
    provides = ("predicted_e_hull",)
    requires = ()
    rank_column = "predicted_e_hull"

    def __init__(self, scores):
        self.scores = scores

    def apply(self, cohort):
        kept = cohort.kept_indices()
        cohort.record("filter", self.name, passed=kept, columns={
            "predicted_e_hull": pd.Series({i: self.scores[i] for i in kept})})

    def describe(self):
        return {"component": self.name}


def test_a_ranked_budget_keeps_the_best_survivors_of_every_filter(tmp_path):
    import numpy as np

    genes = [{"group": 225, "species": ["Na"], "numIons": [4], "sites": [["4a"]],
              "serial": i} for i in range(8)]
    scores = {0: 0.5, 1: -0.4, 2: -0.3, 3: np.nan, 4: -0.2, 5: 0.1, 6: -0.9, 7: 0.0}
    engagement = Engagement(
        roe=TORPEDO_RUN, source=FakeSource(genes), sampler=FakeSampler(),
        filters={"energy": RankingFilter(scores),
                 # The screen removes the best-ranked gene, as a known one would be.
                 "screen": FakeFilter("screen", "screen", keep={0, 1, 2, 3, 4, 5, 7})})
    cohort = engagement.run(8, tmp_path, target_engaged=3)
    assert cohort.kept_indices() == [1, 2, 4], (
        "the three best of what survived the screen, not the three best overall")
    budget = cohort.history[-1]
    assert budget.component == "budget"
    assert budget.detail["selection_cut"] == pytest.approx(-0.2)
    assert cohort.n_sampled == 8, "every draw was needed to rank, so none is dropped"


def test_a_ranked_budget_ranks_undecided_genes_last_and_keeps_a_short_cohort(tmp_path):
    import numpy as np

    genes = [{"group": 225, "species": ["Na"], "numIons": [4], "sites": [["4a"]],
              "serial": i} for i in range(3)]
    engagement = Engagement(
        roe=TORPEDO_RUN, source=FakeSource(genes), sampler=FakeSampler(),
        filters={"energy": RankingFilter({0: np.nan, 1: 0.3, 2: 0.2}),
                 "screen": FakeFilter("screen", "screen", keep={0, 1, 2})})
    assert engagement.run(3, tmp_path / "a", target_engaged=2).kept_indices() == [1, 2]
    assert engagement.run(3, tmp_path / "b", target_engaged=5,
                          max_rounds=1).kept_indices() == [0, 1, 2]


def test_rank_selection_leaves_everything_decided_for_the_budget():
    import numpy as np

    cohort = _cohort_of(4)
    gene_filter = CannedHullFilter({0: 0.9, 1: -0.2, 2: np.nan, 3: 3.0}, select="rank",
                                   margin=None)
    gene_filter.apply(cohort)
    assert cohort.kept_indices() == [0, 1, 2, 3]
    assert gene_filter.rank_column == "predicted_e_hull"

    cohort = _cohort_of(4)
    CannedHullFilter({0: 0.9, 1: -0.2, 2: np.nan, 3: 3.0}, select="rank",
                     margin=1.0, on_missing_hull="drop").apply(cohort)
    assert cohort.kept_indices() == [0, 1]


def test_a_plan_file_can_steer_a_pool_run_without_a_model(tmp_path):
    from wyckoff_transformer.roe.cli import build_parser

    args = build_parser().parse_args([
        "run", "torpedo-run", "--genes", str(tmp_path / "g.json.gz"),
        "--system-plan", str(tmp_path / "plan.json"), "--output-dir", str(tmp_path),
        "--energy-select", "rank", "--energy-hull", "joint", "--energy-basis", "corrected",
        "--residuals", str(tmp_path)])
    assert args.energy_hull == "joint" and args.hull_margin is None
    draw = build_parser().parse_args([
        "draw", "--model-path", "m", "--output-dir", "o", "--n-genes", "10",
        "--system-prior", "p.npz", "--closure", "--n-targets", "3"])
    assert draw.genes is None and draw.closure and draw.n_targets == 3
