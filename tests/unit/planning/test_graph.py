import pytest

from orchestrator.planning.graph import (
    DependencyCycleError,
    UnknownDependencyError,
    parallel_groups,
)


def test_an_empty_plan_has_no_groups() -> None:
    assert parallel_groups({}) == []


def test_sequential_steps_run_one_after_another() -> None:
    assert parallel_groups({"step_1": [], "step_2": ["step_1"], "step_3": ["step_2"]}) == [
        ["step_1"],
        ["step_2"],
        ["step_3"],
    ]


def test_independent_steps_share_one_group() -> None:
    assert parallel_groups({"step_1": [], "step_2": [], "step_3": []}) == [
        ["step_1", "step_2", "step_3"]
    ]


def test_a_diamond_runs_its_two_middle_steps_in_parallel() -> None:
    #      step_1
    #     /      \
    #  step_2  step_3
    #     \      /
    #      step_4
    assert parallel_groups(
        {
            "step_1": [],
            "step_2": ["step_1"],
            "step_3": ["step_1"],
            "step_4": ["step_2", "step_3"],
        }
    ) == [["step_1"], ["step_2", "step_3"], ["step_4"]]


def test_a_complex_graph_waits_for_the_slowest_dependency() -> None:
    # The blog-post shape: two research branches of different depth, then writing and review.
    # step_6 needs step_5 (group 2) and step_2 (group 1), so it lands in group 3.
    assert parallel_groups(
        {
            "step_1": [],
            "step_2": [],
            "step_3": ["step_1"],
            "step_4": [],
            "step_5": ["step_3", "step_4"],
            "step_6": ["step_5", "step_2"],
            "step_7": ["step_6"],
        }
    ) == [["step_1", "step_2", "step_4"], ["step_3"], ["step_5"], ["step_6"], ["step_7"]]


def test_steps_keep_plan_order_within_a_group_not_alphabetical_order() -> None:
    # step_10 is written before step_2 here; sorting by id would put them the other way round.
    groups = parallel_groups({"step_10": [], "step_2": [], "step_3": ["step_10", "step_2"]})
    assert groups == [["step_10", "step_2"], ["step_3"]]


def test_steps_unlocked_together_are_ordered_by_plan_position_not_unlock_order() -> None:
    # Finishing group 0 in order unlocks step_4 (via step_1) before step_3 (via step_2);
    # the next group must still list them in plan order.
    groups = parallel_groups(
        {"step_1": [], "step_2": [], "step_3": ["step_2"], "step_4": ["step_1"]}
    )
    assert groups == [["step_1", "step_2"], ["step_3", "step_4"]]


def test_a_dependency_listed_twice_counts_once() -> None:
    assert parallel_groups({"step_1": [], "step_2": ["step_1", "step_1"]}) == [
        ["step_1"],
        ["step_2"],
    ]


def test_a_cycle_is_reported_with_the_steps_that_form_it() -> None:
    with pytest.raises(DependencyCycleError) as caught:
        parallel_groups({"step_1": ["step_3"], "step_2": ["step_1"], "step_3": ["step_2"]})

    error = caught.value
    assert error.cycle == ("step_1", "step_3", "step_2", "step_1")
    assert str(error) == (
        "dependency cycle (each step depends on the next): step_1 -> step_3 -> step_2 -> step_1"
    )


def test_a_cycle_names_only_its_own_steps_but_lists_everything_it_blocks() -> None:
    # step_1 is fine; step_2 <-> step_3 is the cycle; step_4 is merely stuck behind it.
    with pytest.raises(DependencyCycleError) as caught:
        parallel_groups(
            {
                "step_1": [],
                "step_2": ["step_1", "step_3"],
                "step_3": ["step_2"],
                "step_4": ["step_3"],
            }
        )

    assert caught.value.cycle == ("step_2", "step_3", "step_2")
    assert caught.value.unresolved == ("step_2", "step_3", "step_4")


def test_a_step_that_depends_on_itself_is_a_cycle_of_one() -> None:
    with pytest.raises(DependencyCycleError) as caught:
        parallel_groups({"step_1": [], "step_2": ["step_2"]})

    assert caught.value.cycle == ("step_2", "step_2")


def test_a_dependency_on_a_missing_step_is_rejected_not_ignored() -> None:
    with pytest.raises(UnknownDependencyError) as caught:
        parallel_groups({"step_1": [], "step_2": ["step_1", "step_9", "step_8"]})

    assert caught.value.missing == {"step_2": ("step_8", "step_9")}
    assert "step_2 -> step_8, step_9" in str(caught.value)
