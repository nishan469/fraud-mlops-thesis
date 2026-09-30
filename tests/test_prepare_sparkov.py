import numpy as np
import pandas as pd
import pytest

from prepare_sparkov import card_history_features, haversine_km


def frame(rows):
    df = pd.DataFrame(rows, columns=["cc_num", "ts", "amt"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def test_card_features_use_only_earlier_transactions():
    df = frame([
        (1, "2020-01-01 10:00", 10.0),
        (2, "2020-01-01 11:00", 99.0),     # other card: must not leak into card 1
        (1, "2020-01-01 12:00", 30.0),
        (1, "2020-01-02 11:00", 200.0),    # 23 h after the 12:00 one, 25 h after the first
        (1, "2020-02-15 09:00", 20.0),     # more than 30 days after all earlier ones
    ])
    f = card_history_features(df)
    card1 = f[df["cc_num"] == 1]
    assert card1["card_txns_24h"].tolist() == [0, 1, 1, 0]
    assert card1["card_spend_24h"].tolist() == [0, 10, 30, 0]
    assert card1["card_txns_30d"].tolist() == [0, 1, 2, 0]
    assert np.isnan(card1["amt_vs_card_mean"].iloc[0])           # no history yet
    assert card1["amt_vs_card_mean"].iloc[2] == pytest.approx(200 / 20)
    assert card1["card_secs_since_prev"].iloc[1] == 2 * 3600
    assert f.loc[df["cc_num"] == 2, "card_txns_24h"].item() == 0


def test_features_ignore_the_current_transaction_and_its_label():
    """Changing a transaction's own amount must not change its history features."""
    base = frame([(1, "2020-01-01 10:00", 10.0), (1, "2020-01-01 11:00", 20.0)])
    bumped = base.copy()
    bumped.loc[1, "amt"] = 5000.0
    a, b = card_history_features(base), card_history_features(bumped)
    cols = ["card_txns_24h", "card_spend_24h", "card_txns_30d", "card_secs_since_prev"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_haversine_known_distance():
    # London to Paris is about 344 km
    assert haversine_km(51.5074, -0.1278, 48.8566, 2.3522) == pytest.approx(344, abs=5)
