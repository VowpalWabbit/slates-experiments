import slates


def test_combine_float_actions():
    assert slates.combine_float_actions([1, 2], [1, 2, 3], [1]) == (["x=1 y=1 z=1", "x=1 y=2 z=1", "x=1 y=3 z=1",
                                                                     "x=2 y=1 z=1", "x=2 y=2 z=1", "x=2 y=3 z=1"],  [(1, 1, 1), (1, 2, 1), (1, 3, 1), (2, 1, 1), (2, 2, 1), (2, 3, 1)])


def test_combine_float_actions_categorical():
    assert slates.combine_float_actions_categorical([1, 2], [1, 2, 3], [1]) == (["x=1,y=1,z=1", "x=1,y=2,z=1", "x=1,y=3,z=1",
                                                                                 "x=2,y=1,z=1", "x=2,y=2,z=1", "x=2,y=3,z=1"],  [(1, 1, 1), (1, 2, 1), (1, 3, 1), (2, 1, 1), (2, 2, 1), (2, 3, 1)])


def test_slate_pred_conv():
    before = [[(1, 0.8500000238418579), (2, 0.05000000074505806), (0, 0.05000000074505806), (3, 0.05000000074505806)], [
        (5, 0.8666666746139526), (6, 0.06666667014360428), (4, 0.06666667014360428)], [(8, 0.9000000357627869), (7, 0.10000000149011612)]]
    after = [[(1, 0.8500000238418579), (2, 0.05000000074505806), (0, 0.05000000074505806), (3, 0.05000000074505806)], [
        (1, 0.8666666746139526), (2, 0.06666667014360428), (0, 0.06666667014360428)], [(1, 0.9000000357627869), (0, 0.10000000149011612)]]
    assert slates.slate_pred_conv(before) == after


def test_create_slates_example_no_outcome():
    shared = "shared features"
    action_sets = [["a", "b"], ["c", "d"]]
    example_strings = slates.create_slates_example(
        None, shared, action_sets, debug=True)
    assert example_strings == [
        "ccb shared |User shared features",
        "ccb action |Action a",
        "ccb action |Action b",
        "ccb action |Action c",
        "ccb action |Action d",
        "ccb slot 0,1 |Slot slot_id=0 constant",
        "ccb slot 2,3 |Slot slot_id=1 constant",
    ]


def test_create_slates_example_with_outcome():
    shared = "shared features"
    action_sets = [["a", "b"], ["c", "d"]]
    outcomes = [(0, 0.25, 0.8), (1, 0.5, 0.75)]
    example_strings = slates.create_slates_example(
        None, shared, action_sets, outcomes, debug=True)
    assert example_strings == [
        "ccb shared |User shared features",
        "ccb action |Action a",
        "ccb action |Action b",
        "ccb action |Action c",
        "ccb action |Action d",
        "ccb slot 0:0.25:0.8 0,1 |Slot slot_id=0 constant",
        "ccb slot 3:0.5:0.75 2,3 |Slot slot_id=1 constant",
    ]


def test_create_combinatorial_cb_example_no_outcome():
    shared = "shared features"
    actions = ["a,c", "a,d", "b,c", "b,d"]
    outcome = (2, 0.5, 0.8)
    example_strings = slates.create_cb_example(
        None, shared, actions, outcome, debug=True)
    assert example_strings == [
        "shared |User shared features",
        "|Action a,c",
        "|Action a,d",
        "2:0.5:0.8 |Action b,c",
        "|Action b,d",
    ]


def test_create_combinatorial_cb_example_with_outcome():
    shared = "shared features"
    actions = ["a,c", "a,d", "b,c", "b,d"]
    example_strings = slates.create_cb_example(
        None, shared, actions, debug=True)
    assert example_strings == [
        "shared |User shared features",
        "|Action a,c",
        "|Action a,d",
        "|Action b,c",
        "|Action b,d",
    ]

def test_combine():
    assert slates.combine([[1,2],[3]], ["x", "y"]) == ["x=1 y=3", "x=2 y=3"]
    assert slates.combine([[1,2],[3]], ["x", "y"], fmt_str="{}={},{}") == ["x=1,y=3", "x=2,y=3"]
    