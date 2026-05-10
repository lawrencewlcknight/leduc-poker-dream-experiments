
def test_import_shared_modules():
    import dream_poker.constants  # noqa: F401
    import dream_poker.experiment_utils  # noqa: F401
    import dream_poker.networks  # noqa: F401
    import dream_poker.replay  # noqa: F401
    import dream_poker.seeding  # noqa: F401


def test_replay_buffer_adds_items():
    from dream_poker.replay import ReservoirBuffer

    buf = ReservoirBuffer(10)
    for i in range(20):
        buf.add(i)
    assert len(buf) == 10
    assert buf.add_calls == 20
