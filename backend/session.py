"""
Sign session state machine.

Turns a stream of per-frame classifier outputs into confirmed words and
finished sentences. It is pure Python with an injectable clock so it can be
unit-tested without a camera, a model, or a WebSocket.

Rules
  * Hold:   the same sign at >= threshold confidence for HOLD_SECONDS
            (brief dropouts up to HOLD_DROPOUT_SECONDS are tolerated).
  * Repeat: the same sign is not confirmed twice in a row unless the hands
            leave the frame for REPEAT_RESET_SECONDS in between.
  * Pause:  hands absent for PAUSE_SECONDS with words pending -> sentence.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from backend import config

NONE_SIGN = "NONE"


@dataclass
class FrameOutcome:
    hands_detected: int
    top_prediction: str
    confidence: float
    hold_progress: float
    pause_progress: float
    words: List[str]
    candidate: Optional[str] = None
    confirmed_word: Optional[str] = None
    sentence_words: Optional[List[str]] = None  # set when a pause finishes a sentence


@dataclass
class SignSession:
    threshold: float = config.CONFIDENCE_THRESHOLD
    hold_seconds: float = config.HOLD_SECONDS
    dropout_seconds: float = config.HOLD_DROPOUT_SECONDS
    pause_seconds: float = config.PAUSE_SECONDS
    repeat_reset_seconds: float = config.REPEAT_RESET_SECONDS
    max_words: int = config.MAX_WORDS
    clock: Callable[[], float] = time.monotonic

    words: List[str] = field(default_factory=list)
    _candidate: Optional[str] = None
    _candidate_since: float = 0.0
    _last_good_at: float = 0.0
    _last_confirmed: Optional[str] = None
    _hands_last_seen: Optional[float] = None
    # The pause auto-send only applies to signed words; tapped words wait for
    # an explicit finalize so a slow tapper is never cut off.
    _pause_armed: bool = False

    # Commands -----------------------------------------------------------

    def reset(self) -> None:
        self.words.clear()
        self._candidate = None
        self._last_confirmed = None
        self._pause_armed = False

    def remove_word(self, index: int) -> bool:
        if 0 <= index < len(self.words):
            self.words.pop(index)
            self._last_confirmed = self.words[-1] if self.words else None
            return True
        return False

    def add_word(self, sign: str) -> bool:
        """Manual entry (tap-to-sign fallback)."""
        if sign == NONE_SIGN or len(self.words) >= self.max_words:
            return False
        self.words.append(sign)
        self._last_confirmed = sign
        self._pause_armed = False
        return True

    def finalize(self) -> Optional[List[str]]:
        """Finish the sentence now instead of waiting for the pause."""
        if not self.words:
            return None
        finished = list(self.words)
        self.reset()
        return finished

    # Frame processing --------------------------------------------------

    def process(self, hands_detected: int, sign: str, confidence: float) -> FrameOutcome:
        now = self.clock()
        if hands_detected == 0:
            return self._process_absent(now)

        self._hands_last_seen = now
        good = sign != NONE_SIGN and confidence >= self.threshold

        if good and sign == self._candidate:
            self._last_good_at = now
        elif good:
            # A different confident sign takes over only once the old one has
            # really gone (tolerates single misclassified frames).
            if (
                self._candidate is None
                or self._candidate == self._last_confirmed
                or now - self._last_good_at > self.dropout_seconds
            ):
                self._candidate = sign
                self._candidate_since = now
                self._last_good_at = now
        elif self._candidate is not None and now - self._last_good_at > self.dropout_seconds:
            self._candidate = None

        hold = 0.0
        confirmed = None
        if self._candidate is not None:
            held_for = now - self._candidate_since
            hold = min(1.0, held_for / self.hold_seconds) if self.hold_seconds > 0 else 1.0
            if self._candidate == self._last_confirmed:
                hold = 0.0  # already confirmed; waiting for a change
            elif hold >= 1.0 and len(self.words) < self.max_words:
                confirmed = self._candidate
                self.words.append(confirmed)
                self._last_confirmed = confirmed
                self._pause_armed = True
                hold = 0.0

        return FrameOutcome(
            hands_detected=hands_detected,
            top_prediction=sign,
            confidence=confidence,
            hold_progress=round(hold, 3),
            pause_progress=0.0,
            words=list(self.words),
            candidate=self._candidate,
            confirmed_word=confirmed,
        )

    def _process_absent(self, now: float) -> FrameOutcome:
        self._candidate = None
        absent_for = 0.0 if self._hands_last_seen is None else now - self._hands_last_seen
        if absent_for >= self.repeat_reset_seconds:
            self._last_confirmed = None

        pause = 0.0
        sentence = None
        if self.words and self._pause_armed:
            pause = min(1.0, absent_for / self.pause_seconds) if self.pause_seconds > 0 else 1.0
            if pause >= 1.0:
                sentence = self.finalize()
                pause = 0.0

        return FrameOutcome(
            hands_detected=0,
            top_prediction=NONE_SIGN,
            confidence=0.0,
            hold_progress=0.0,
            pause_progress=round(pause, 3),
            words=list(self.words),
            sentence_words=sentence,
        )
