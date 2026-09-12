"""Headless tests of the Streamlit app itself.

`AppTest` runs each page with no browser and collects every exception the run
raised. It catches the whole class of failures the maths tests cannot see:
bad widget keys, state written after its widget was created, stale API calls,
charts built from a frame that does not have the column.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

#: Absolute, because Streamlit changed what a relative AppTest path is relative
#: to: the working directory up to 1.58, the calling test file from 1.63.
APP = str(Path(__file__).resolve().parent.parent / "app.py")

PAGES = ["home", "pricer", "surfaces", "builder", "hedging", "volsmile", "lessons"]

SCRIPT = """
import streamlit as st
from optlab.strategies import Market
st.session_state.market = Market(spot=100.0, rate=0.04, div_yield=0.0, vol=0.20)
from views import {module}
{module}.render()
"""


def run_page(module: str, timeout: int = 120) -> AppTest:
    at = AppTest.from_string(SCRIPT.format(module=module), default_timeout=timeout)
    return at.run()


@pytest.mark.parametrize("module", PAGES)
def test_page_runs_without_exceptions(module):
    at = run_page(module)
    assert not at.exception, f"{module} raised: {[e.value for e in at.exception]}"


@pytest.mark.parametrize("module", PAGES)
def test_page_renders_something(module):
    at = run_page(module)
    assert at.title or at.markdown or at.header


def test_whole_app_boots():
    """The real entry point, navigation and shared sidebar included."""
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


# --- driving the pages ----------------------------------------------------


def test_pricer_reacts_to_a_different_greek():
    at = run_page("pricer")
    picker = next(s for s in at.selectbox if s.key == "pr_greek")
    picker.select("gamma").run()
    assert not at.exception
    assert at.session_state["pr_greek"] == "gamma"


def test_pricer_handles_both_option_types():
    at = run_page("pricer")
    for kind in ("put", "call"):
        at.session_state["pr_kind"] = kind
        at.run()
        assert not at.exception


def test_pricer_survives_an_extreme_maturity():
    at = run_page("pricer")
    next(s for s in at.slider if s.key == "pr_days").set_value(1).run()
    assert not at.exception


def test_surfaces_renders_every_greek():
    from optlab import bs

    at = run_page("surfaces")
    picker = next(s for s in at.selectbox if s.key == "sf_greek")
    for greek in bs.GREEK_FUNCS:
        picker.select(greek).run()
        assert not at.exception, f"{greek} map raised"


def test_surfaces_renders_every_greek_in_3d():
    from optlab import bs

    at = run_page("surfaces")
    at.session_state["sf_view"] = "3-D surface"
    picker = next(s for s in at.selectbox if s.key == "sf_greek")
    for greek in bs.GREEK_FUNCS:
        picker.select(greek).run()
        assert not at.exception, f"{greek} surface raised"


def test_surface_view_survives_the_peak_cap():
    at = run_page("surfaces")
    at.session_state["sf_view"] = "3-D surface"
    at.session_state["sf_greek"] = "gamma"
    at.run()
    assert not at.exception
    at.session_state["sf_cap"] = True
    at.run()
    assert not at.exception


def test_switching_between_map_and_surface_is_clean():
    at = run_page("surfaces")
    for view in ("3-D surface", "Flat map", "3-D surface"):
        at.session_state["sf_view"] = view
        at.run()
        assert not at.exception, f"switching to {view!r} raised"


def test_builder_loads_every_preset():
    from optlab.strategies import PRESETS

    at = run_page("builder")
    picker = next(s for s in at.selectbox if s.key == "sb_preset")
    for preset in PRESETS:
        picker.select(preset).run()
        assert not at.exception, f"{preset} raised"


def test_builder_time_slider_moves_toward_expiry():
    at = run_page("builder")
    slider = next(s for s in at.slider if s.key == "sb_horizon")
    slider.set_value(slider.max).run()
    assert not at.exception


def test_hedging_lab_runs_a_simulation():
    at = run_page("hedging")
    assert not at.exception
    assert at.metric, "the hedging lab should report summary metrics"


def test_hedging_lab_path_picker_covers_every_option():
    from optlab import hedge as hedge_mod

    at = run_page("hedging")
    for choice in list(hedge_mod.NOTABLE) + ["Pick by number"]:
        at.session_state["hg_path_pick"] = choice
        at.run()
        assert not at.exception, f"path choice {choice!r} raised"


def test_hedging_lab_accepts_an_explicit_path_number():
    at = run_page("hedging")
    at.session_state["hg_path_pick"] = "Pick by number"
    at.session_state["hg_path_index"] = 17
    at.run()
    assert not at.exception
    assert any("Path 17" in m.value for m in at.markdown), "the page should name the path"


def test_hedging_lab_with_costs_and_a_band():
    at = run_page("hedging")
    at.session_state["hg_cost"] = 5.0
    at.session_state["hg_band"] = 0.1
    at.session_state["hg_steps"] = 25
    at.run()
    assert not at.exception


def test_smile_shapes_all_render():
    from optlab import smile

    at = run_page("volsmile")
    picker = next(s for s in at.selectbox if s.key == "vs_shape")
    for shape in smile.SHAPES:
        picker.select(shape).run()
        assert not at.exception, f"{shape} raised"


def test_implied_vol_rejects_an_impossible_price():
    """A price below intrinsic must warn, not crash or print a bogus vol."""
    at = run_page("volsmile")
    at.session_state["vs_iv_strike"] = 50.0
    at.session_state["vs_iv_price"] = 0.01  # far below intrinsic for a 50-strike call
    at.run()
    assert not at.exception
    assert at.warning, "an unreachable price should raise a visible warning"


def test_lessons_reveal_flow():
    at = run_page("lessons")
    at.session_state["lesson_state"] = {
        "sqrt_time": {"choice": 1, "revealed": True},
        "gamma_theta": {"choice": 0, "revealed": True},
    }
    at.run()
    assert not at.exception
    assert at.success, "a correct answer should be marked correct"
    assert at.error, "a wrong answer should be marked wrong"


def test_every_lesson_explanation_runs():
    """Reveal all seven at once - each explanation recomputes from the model."""
    from views.lessons import LESSONS

    at = run_page("lessons", timeout=240)
    at.session_state["lesson_state"] = {
        l["id"]: {"choice": l["answer"], "revealed": True} for l in LESSONS
    }
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_revealing_without_choosing_does_not_crash():
    """A revealed-but-unanswered lesson must still render its explanation."""
    at = run_page("lessons")
    at.session_state["lesson_state"] = {"sqrt_time": {"choice": None, "revealed": True}}
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_reveal_is_disabled_until_an_answer_is_picked():
    at = run_page("lessons")
    reveals = [b for b in at.button if (b.key or "").startswith("reveal_")]
    assert reveals, "each unanswered lesson should offer a reveal button"
    assert all(b.disabled for b in reveals), "reveal must wait for a guess"


def test_lesson_answers_are_all_in_range():
    from views.lessons import LESSONS

    ids = [l["id"] for l in LESSONS]
    assert len(ids) == len(set(ids)), "lesson ids must be unique"
    for lesson in LESSONS:
        assert 0 <= lesson["answer"] < len(lesson["options"])
        assert callable(lesson["explain"])
