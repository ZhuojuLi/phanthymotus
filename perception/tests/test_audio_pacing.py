"""The delivery clock must preserve PCM timing across partial frames and EOF."""
import pytest

from utils.audio_pacing import AudioLeadPacer


def send(pacer, size, now):
    now += pacer.delay(size, now)
    pacer.sent(size, now)
    return now


def test_last_partial_frame_uses_its_actual_sample_duration():
    pacer = AudioLeadPacer(1.0)
    now = 100.0
    for _ in range(154):
        now = send(pacer, 3200, now)
    now = send(pacer, 2400, now)
    # 15.475s of audio is delivered in 14.475s, retaining exactly 1s of lead.
    assert now == pytest.approx(114.475)


def test_consecutive_short_utterances_share_one_lead_budget():
    pacer = AudioLeadPacer(1.0)
    now = 100.0
    samples_sent = 0
    for _ in range(20):
        for size in [3200, 3200, 1600]:
            now = send(pacer, size, now)
            samples_sent += size
            assert 0 < 100 + samples_sent / 32000 - now <= 1.000001
    assert now == pytest.approx(104.0)


def test_idle_time_reanchors_without_accumulating_burst_credit():
    pacer = AudioLeadPacer(1.0)
    now = send(pacer, 3200, 100.0)
    now = send(pacer, 3200, now + 60)
    assert now == 160.0
    assert pacer.delay(32000, now) == pytest.approx(.1)


def test_a_frame_larger_than_the_budget_fails_instead_of_waiting_forever():
    pacer = AudioLeadPacer(.05)
    with pytest.raises(ValueError, match='must fit'):
        pacer.delay(3200, 100.0)


@pytest.mark.parametrize('lead', [0, -1, float('inf'), float('nan')])
def test_invalid_lead_is_rejected(lead):
    with pytest.raises(ValueError, match='finite and positive'):
        AudioLeadPacer(lead)
