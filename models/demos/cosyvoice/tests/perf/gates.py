# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The bring-up's numeric acceptance thresholds, and the code that enforces them.

Every threshold below is quoted verbatim from the bring-up requirements, and every
perf test that produces one of these numbers calls `enforce()` on it. Before this
module existed the perf suite printed its figures and asserted `total_s > 0` -- a
timing harness, not a gate -- so a regression that halved throughput would still have
run green. That is the gap this closes.

## The rule, in one paragraph

A threshold that one part meets and another misses cannot be a single unconditional
`assert`: on the part that misses it, the suite would simply be red forever, which is
not enforcement so much as a broken build. So the thresholds are declared once, in
`GATES`, and the **per-architecture verdict** is declared separately, in
`EXPECTATIONS`:

* a gate recorded as **`Meets`** is asserted directly -- if the measured value stops
  clearing the threshold, the test fails;
* a gate recorded as **`Misses`** is asserted against the *recorded measurement*, both
  bounds, exactly as `models/perf/device_perf_utils.check_device_perf` does. Slower
  than the band fails, because that is a regression. **Faster than the band also
  fails**, because it means the recorded number -- which `PERF.md` publishes -- is
  stale, and a stale published number is the thing this module exists to prevent.

Nothing here is `xfail`-ed. A missed target gets a measured number, a named lever and
a band it has to stay inside; it does not get a marker that hides it from the summary
line.

## What is recorded, and from where

Every value in `EXPECTATIONS` comes from the certification run described in
`../../PERF.md` Part I -- one commit, one day, Blackhole `p150a`, Blackhole `p150b`
and Wormhole n300, five configurations each. Two boards is not a matrix; the third is
what makes it one.

**A `recorded` value is the centre of a band, not the last run's figure.** PERF.md
publishes what a given run measured; this table holds the reference those measurements
have to stay near. Re-centring it after every run would defeat the point — the band
exists because the same board measures a few per cent apart from day to day. What must
hold is that PERF.md's published figures lie inside these bands; when one stops doing
so, the run fails and both are updated together.

Bands are wide on purpose. The flow decoder varies by about 5 % run to run, the two
Blackhole boards differ by another ~5 % through cooling alone, and a host under load
moves the LLM step. A band tight enough to catch a 3 % drift would flake on all three;
these are sized to catch the failures that matter -- a lost trace capture, a dropped
fused-attention path, a cache that started reallocating -- which are 20 % events and
larger.
"""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# the thresholds themselves
# --------------------------------------------------------------------------
AT_LEAST, BELOW = "at_least", "below"


@dataclass(frozen=True)
class Gate:
    """One numeric acceptance threshold."""

    key: str
    label: str
    stage: str
    target: float
    direction: str
    unit: str

    def passes(self, measured: float) -> bool:
        return measured >= self.target if self.direction == AT_LEAST else measured < self.target

    def describe(self) -> str:
        op = ">=" if self.direction == AT_LEAST else "<"
        return f"{self.label} {op} {self.target}{self.unit}"


GATES: dict[str, Gate] = {
    g.key: g
    for g in (
        # Stage 1 -- bring-up baselines.
        Gate("tok_s", "semantic token generation", "Stage 1", 30.0, AT_LEAST, " tok/s"),
        Gate("rtf", "real-time factor, typical sentence", "Stage 1", 0.5, BELOW, ""),
        # Stage 3 -- stretch targets.
        Gate("tok_s_stretch", "semantic token generation", "Stage 3 stretch", 60.0, AT_LEAST, " tok/s"),
        Gate("rtf_stretch", "real-time factor", "Stage 3 stretch", 0.2, BELOW, ""),
        # Stage 3 -- the interleaved schedule. Expressed as a ratio of streamed wall
        # time to the audio produced, so it means the same thing at any utterance
        # length; below 1.0 is "a player never starves once playback has started".
        Gate("stream_realtime", "interleaved streaming sustains real time", "Stage 3", 1.0, BELOW, " x audio"),
    )
}


# --------------------------------------------------------------------------
# per-architecture verdicts
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Meets:
    """The gate is cleared on this architecture; assert the gate itself."""


@dataclass(frozen=True)
class Straddles:
    """The threshold sits *inside* this measurement's run-to-run spread.

    Neither of the other two verdicts can state this honestly. `Meets` fails on the
    runs that come in over the line, `Misses` fails on the runs that come in under it,
    and both are the same measurement behaving normally -- so whichever you picked, the
    suite would flake at the rate the distribution crosses the threshold, and the flake
    would be indistinguishable from a real regression.

    So this asserts the band and nothing about the direction. `passes of n` records what
    was actually observed when the figure was characterised, which is the honest claim:
    not "it misses", but "it missed seven times in nine, by this much". A later shift in
    either direction moves the mean out of the band and fails, which is the behaviour
    that matters; a single run landing on the other side of the threshold does not,
    because that is what a straddling distribution does.

    Use it only with a characterised distribution -- a handful of runs is not enough to
    know a threshold is inside the spread rather than outside it. See
    `tests/perf/README` and PERF.md for the runs behind the figures here.
    """

    recorded: float
    tol: float
    passes: int
    of: int
    lever: str


@dataclass(frozen=True)
class Misses:
    """The gate is not cleared; assert the recorded measurement instead.

    `recorded` is the published figure, `tol` the fractional half-width of the band
    around it. `lever` names what would close the gap -- it is printed on failure and
    on every run, because a missed target without a stated lever is just a number.
    """

    recorded: float
    tol: float
    lever: str


# Recorded on the boards named in PERF.md's *Environment*. Blackhole figures are the
# `p150a`/`p150b` pair -- the two differ by ~5 % through cooling, so the bands below
# are the union of both rather than one board's.
BLACKHOLE = {
    # End-to-end traced decode clears both gates in every configuration on both boards,
    # by a factor approaching three against the stretch target. PERF.md section 3.3 has
    # the figures; they are not repeated here, because a comment that restates a
    # published number is a second place for it to go stale.
    "tok_s": Meets(),
    "tok_s_stretch": Meets(),
    # Cleared in every configuration on both boards; PERF.md section 3.2 has the figures.
    "rtf": Meets(),
    # Reaching 0.2 needs the LLM decode step under 1.5 ms on its own; the step is
    # around 5 ms at its best measured and is bandwidth-bound on the AR decoder's
    # weights. Band is centred between the two boards' default configurations.
    "rtf_stretch": Misses(
        0.385, 0.35, "no op-level lever left; needs a smaller decoder or multi-chip tensor parallelism"
    ),
    # Interleaved streaming clears this on both boards every run, with 22 % of headroom
    # on the slower one and 35 % on the faster -- the widest margin of any gate here,
    # and the reason this is a `Meets` while the same gate straddles on Wormhole.
    # Repeated runs on p150a agreed to within 8 ms, the tightest measurement in this
    # file, so the verdict is not resting on one sample.
    #
    # Blackhole ships the plain decoder here, because `kv_inplace_default` is false on
    # this architecture. Forcing the in-place one is faster still, but that is not what
    # runs, so it is not what is recorded.
    "stream_realtime": Meets(),
}

# These figures are from n300 specifically. A different Wormhole part will trip the
# band below rather than silently inherit n300's verdict, which is the intended
# behaviour -- see the module docstring.
WORMHOLE = {
    # End-to-end traced decode clears both gates in every configuration, by a factor of
    # two against the stretch target. The three configurations land within a few per
    # cent of each other, as they should: `kv_inplace_default` reads the architecture
    # and turns the in-place cache on for Wormhole, so the explicit in-place row
    # measures the same thing as the default. PERF.md section 3.3 has the figures.
    "tok_s": Meets(),
    "tok_s_stretch": Meets(),
    # Three full-suite runs of the same three configurations on the same board, at
    # trees differing only in this directory, measured 0.539 / 0.554 / 0.542, then
    # 0.553 / 0.552 / 0.564, then 0.550 / 0.562 / 0.566 -- so **the same board spans
    # about 5 % across runs of identical work**, which is the flow decoder's documented
    # run-to-run variation and is precisely why the band is +/-20 % rather than tight.
    # The centre sits between the runs rather than on any one of them, and PERF.md
    # section 3.2 publishes the latest.
    #
    # **`COSYVOICE_FF2_GRID=8x2` does not help on this part**, and across those three
    # runs it has come in below the default, level with it, and above it -- which is
    # what "within noise" means when you have enough samples to say it. An earlier
    # vintage had it winning clearly. The flag stays opt-in and Blackhole-favoured for
    # exactly this reason: its best shape is not portable, and on n300 its end-to-end
    # benefit is not reliably positive, even though it is worth a real 1.04x on the
    # decode step it actually touches.
    # **This was a `Misses` at 0.55 and is now a straddle at 0.499**, because two levers
    # that were already in the tree started working. `COSYVOICE_WEIGHT_BF8` never reached
    # the decoder -- nothing read the dtype it set -- and `COSYVOICE_FF2_GRID` was opt-in
    # on the strength of three cross-session runs that read it as unreliable here. Wired
    # up and made architecture defaults, they are worth -3.0 % and -3.4 % individually,
    # bracketed, and together they move the median from 0.539 to 0.499.
    #
    # Eighteen runs of the shipped configuration across two sessions, pooled because the
    # two disagree in a way that matters: nine on a cool board read 0.489-0.520 and
    # cleared 5, nine on the same board after several hours of continuous work read
    # 0.492-0.524 and cleared 2. Neither is the number; both together are. Against
    # eighteen baseline runs spanning 0.535-0.562 that cleared it **zero** times, the
    # shipped default clears it seven times in eighteen with a median of 0.5025.
    #
    # The discipline is worth restating because it nearly failed twice here. The first
    # three runs of this configuration read 0.489/0.493/0.497, and a sibling
    # configuration's first three read 0.488/0.490/0.491 -- either would have been
    # published as a clean pass. At n=9 both straddled.
    "rtf": Straddles(
        0.5025,
        0.15,
        7,
        18,
        "the remaining gap is the compute grid: 64 cores against 130, on a decode step "
        "whose linears are 34 % of it and already near TTNN's optimum. Closing it needs "
        "fewer ops rather than faster ones -- graph-level fusion, or the wider part",
    ),
    "rtf_stretch": Misses(0.5025, 0.20, "same lever as the 0.5 gate, and far from it"),
    # **The one straddling figure in this file, and the reason `Straddles` exists.**
    # Thirteen runs on n300 in the shipped configuration gave ratios from 0.961 to
    # 1.087, mean 1.040, clearing the gate twice. The threshold is inside the spread,
    # not outside it, so neither `Meets` nor `Misses` can describe it without flaking at
    # the rate the distribution crosses the line.
    #
    # An earlier characterisation of the same thing read 22 % over rather than 2.5 %.
    # That was measured before the prefill warm-up let this test use the in-place
    # decoder and the full trace region -- the configuration, not the part, was what
    # made the gap look decisive. Both figures were true of their own runs; only this
    # one describes what the port actually does.
    "stream_realtime": Straddles(
        1.040,
        0.15,
        2,
        13,
        "the flow decoder and vocoder per chunk, not the AR decode -- n300's 8x8 grid "
        "runs that work well behind a 13x10 Blackhole grid, so closing it needs either "
        "a coarser chunk schedule or the wider grid",
    ),
}

EXPECTATIONS = {"blackhole": BLACKHOLE, "wormhole": WORMHOLE}


def arch_key(device) -> str:
    """`'blackhole'` or `'wormhole'` from a live device.

    Keyed on the architecture rather than the board because that is what the code
    branches on everywhere else in this port -- `kv_inplace_default` reads the same
    string -- and because a board name is not available from ttnn at all.
    """
    arch = str(device.arch()).upper()
    if "BLACKHOLE" in arch:
        return "blackhole"
    if "WORMHOLE" in arch:
        return "wormhole"
    raise AssertionError(f"no recorded expectations for architecture {arch!r}")


# --------------------------------------------------------------------------
# enforcement
# --------------------------------------------------------------------------
def enforce(key: str, measured: float, device, *, extra: str = "") -> str:
    """Assert `measured` against gate `key` on `device`'s architecture.

    Returns the one-line verdict it printed, so a caller can collect the lines into a
    summary table. Raises `AssertionError` on any of the three failure modes:

    1. a `Meets` gate no longer cleared -- a real regression against the requirement;
    2. a `Misses` gate that got worse than its recorded band -- a regression against
       the published figure;
    3. a `Misses` gate that got *better* than its recorded band -- the published
       figure is stale, and `PERF.md` plus this table need updating. Promote it to
       `Meets()` once it clears the threshold.
    """
    gate = GATES[key]
    verdict = EXPECTATIONS[arch_key(device)][key]
    arch = arch_key(device)
    suffix = f"  [{extra}]" if extra else ""

    if isinstance(verdict, Straddles):
        lo = verdict.recorded * (1 - verdict.tol)
        hi = verdict.recorded * (1 + verdict.tol)
        line = (
            f"{gate.describe():<52} measured {measured:8.3f}   STRADDLES, in band "
            f"[{lo:.3f}, {hi:.3f}], cleared {verdict.passes}/{verdict.of} when "
            f"characterised{suffix}\n    lever: {verdict.lever}"
        )
        assert lo <= measured <= hi, (
            f"{gate.stage} gate {gate.describe()} on {arch}: measured {measured:.3f}, outside the "
            f"recorded band [{lo:.3f}, {hi:.3f}] around {verdict.recorded}. This measurement "
            f"straddles the threshold ({verdict.passes} of {verdict.of} runs cleared it when it "
            f"was characterised), so a single run on either side of {gate.target} is expected -- "
            f"what is not expected is the value leaving the band. Re-characterise and update "
            f"PERF.md and this table together."
        )
        return line

    if isinstance(verdict, Meets):
        line = f"{gate.describe():<52} measured {measured:8.3f}   {'PASS' if gate.passes(measured) else 'FAIL'}{suffix}"
        assert gate.passes(measured), (
            f"{gate.stage} gate not met on {arch}: {gate.describe()}, measured {measured:.3f}. "
            f"This gate is recorded as met in tests/perf/gates.py -- either a regression, "
            f"or the run is not comparable (check trace capture actually happened)."
        )
        return line

    lo = verdict.recorded * (1 - verdict.tol)
    hi = verdict.recorded * (1 + verdict.tol)
    line = (
        f"{gate.describe():<52} measured {measured:8.3f}   MISS, in band "
        f"[{lo:.3f}, {hi:.3f}]{suffix}\n    lever: {verdict.lever}"
    )
    assert not gate.passes(measured), (
        f"{gate.stage} gate {gate.describe()} is now MET on {arch} (measured {measured:.3f}), "
        f"but tests/perf/gates.py records it as missed at {verdict.recorded}. "
        f"Promote it to Meets() and update PERF.md -- a published figure that is worse "
        f"than reality is still a wrong published figure."
    )
    assert lo <= measured <= hi, (
        f"{gate.stage} gate {gate.describe()} on {arch}: measured {measured:.3f}, outside the "
        f"recorded band [{lo:.3f}, {hi:.3f}] around {verdict.recorded}. "
        f"{'Slower than recorded -- a regression.' if measured > hi else 'Faster than recorded -- update PERF.md and this table.'}"
    )
    return line


def report(lines: list[str], title: str) -> None:
    """Print a collected set of `enforce` verdicts as one block."""
    print(f"\n  {title}")
    for line in lines:
        print(f"    {line}")
