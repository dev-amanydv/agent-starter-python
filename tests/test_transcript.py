from transcript import TurnAggregator


def _collect():
    turns: list[tuple[str, str, float]] = []
    return turns, lambda role, content, created_at: turns.append(
        (role, content, created_at)
    )


def test_merges_consecutive_same_role_items():
    """A user answer split across several committed items becomes one turn."""
    turns, on_turn = _collect()
    agg = TurnAggregator(on_turn)

    agg.add("user", "I worked on an project, like,", 1.0)
    agg.add("user", "would ask the user about", 2.0)
    agg.add("user", "context of user's resume.", 3.0)
    assert turns == []

    agg.add("assistant", "That sounds relevant.", 4.0)
    assert turns == [
        ("user", "I worked on an project, like, would ask the user about context of user's resume.", 1.0)
    ]


def test_flush_emits_trailing_turn():
    """flush() (called at session end) persists the last open turn."""
    turns, on_turn = _collect()
    agg = TurnAggregator(on_turn)

    agg.add("assistant", "Welcome to the interview.", 1.0)
    agg.add("user", "Hey.", 2.0)
    agg.add("user", "I'm ready.", 3.0)
    agg.flush()

    assert turns == [
        ("assistant", "Welcome to the interview.", 1.0),
        ("user", "Hey. I'm ready.", 2.0),
    ]


def test_created_at_is_turn_start():
    """The merged turn keeps the timestamp of its first item."""
    turns, on_turn = _collect()
    agg = TurnAggregator(on_turn)

    agg.add("user", "part one", 10.0)
    agg.add("user", "part two", 20.0)
    agg.flush()

    assert turns == [("user", "part one part two", 10.0)]


def test_ignores_blank_and_unknown_roles():
    """Empty text and non user/assistant roles are dropped, not buffered."""
    turns, on_turn = _collect()
    agg = TurnAggregator(on_turn)

    agg.add("user", "   ", 1.0)
    agg.add("user", None, 2.0)
    agg.add("system", "ignored", 3.0)
    agg.flush()

    assert turns == []


def test_flush_is_idempotent_when_empty():
    """Flushing with nothing buffered emits no turn."""
    turns, on_turn = _collect()
    agg = TurnAggregator(on_turn)

    agg.flush()
    agg.flush()

    assert turns == []
