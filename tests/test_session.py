"""Unit tests for the time-based sign session state machine."""

from backend.session import SignSession


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_session(**kw):
    clock = FakeClock()
    params = dict(threshold=0.8, hold_seconds=1.0, dropout_seconds=0.2,
                  pause_seconds=1.5, repeat_reset_seconds=0.4, clock=clock)
    params.update(kw)
    return SignSession(**params), clock


def feed(session, clock, sign, conf, seconds, fps=20, hands=1):
    outcomes = []
    for _ in range(int(seconds * fps)):
        clock.advance(1.0 / fps)
        outcomes.append(session.process(hands, sign, conf))
    return outcomes


def test_hold_confirms_after_hold_seconds():
    s, c = make_session()
    out = feed(s, c, "TICKET", 0.95, 0.9)
    assert all(o.confirmed_word is None for o in out)
    assert 0.8 < out[-1].hold_progress < 1.0
    out = feed(s, c, "TICKET", 0.95, 0.2)
    assert [o.confirmed_word for o in out if o.confirmed_word] == ["TICKET"]
    assert s.words == ["TICKET"]


def test_low_confidence_never_confirms():
    s, c = make_session()
    out = feed(s, c, "TICKET", 0.5, 3.0)
    assert s.words == []
    assert all(o.hold_progress == 0 for o in out)


def test_single_frame_dropout_is_tolerated():
    s, c = make_session()
    feed(s, c, "TRAIN", 0.95, 0.5)
    feed(s, c, "WHEN", 0.9, 0.05)  # one misclassified frame
    feed(s, c, "TRAIN", 0.95, 0.6)
    assert s.words == ["TRAIN"]


def test_same_sign_not_repeated_while_held():
    s, c = make_session()
    feed(s, c, "TICKET", 0.95, 4.0)
    assert s.words == ["TICKET"]


def test_same_sign_repeats_after_hands_leave():
    s, c = make_session()
    feed(s, c, "0", 0.95, 1.2)
    feed(s, c, "NONE", 0.0, 0.5, hands=0)
    feed(s, c, "0", 0.95, 1.2)
    assert s.words == ["0", "0"]


def test_pause_finishes_sentence():
    s, c = make_session()
    feed(s, c, "TRAIN", 0.95, 1.2)
    feed(s, c, "WHEN", 0.95, 1.2)
    out = feed(s, c, "NONE", 0.0, 1.6, hands=0)
    finished = [o.sentence_words for o in out if o.sentence_words]
    assert finished == [["TRAIN", "WHEN"]]
    assert s.words == []
    assert max(o.pause_progress for o in out) > 0.9


def test_tapped_words_wait_for_finalize():
    s, c = make_session()
    s.add_word("WATER")
    s.add_word("WHERE")
    out = feed(s, c, "NONE", 0.0, 5.0, hands=0)
    assert all(o.sentence_words is None for o in out)
    assert s.finalize() == ["WATER", "WHERE"]
    assert s.finalize() is None


def test_max_words_enforced():
    s, c = make_session(max_words=2)
    assert s.add_word("1") and s.add_word("2")
    assert not s.add_word("3")
    assert not s.add_word("NONE")


def test_remove_word():
    s, _ = make_session()
    s.add_word("HELP")
    s.add_word("DOCTOR")
    assert s.remove_word(0)
    assert not s.remove_word(5)
    assert s.words == ["DOCTOR"]
