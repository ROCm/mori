from mori.ops.dispatch_combine_v2 import tuning_configs as tc


def test_expected_unique_peers_bounds():
    assert tc.expected_unique_peers(1, 8) == 1.0
    assert 1.0 < tc.expected_unique_peers(8, 8) < 8.0
    assert tc.expected_unique_peers(8, 4) < tc.expected_unique_peers(8, 8)


def test_analytical_schedule_is_legal_and_volume_scaled(monkeypatch):
    monkeypatch.setattr(tc, "_cu_count", lambda: 256)
    monkeypatch.setattr(tc, "_topology", lambda: (0, 90500))
    full = tc.lookup_analytical(8, 7168, 8, "bf16")["schedule"]
    smaller = tc.lookup_analytical(8, 4096, 8, "bf16")["schedule"]

    assert [row[0] for row in full[:-1]] == sorted(
        row[0] for row in full[:-1]
    )
    assert full[-1][0] is None
    assert smaller[0][0] > full[0][0]
    for schedule in (full, smaller):
        for _, disp_block, disp_warp, comb_block, comb_warp in schedule:
            assert 0 < disp_block <= 256
            assert 0 < comb_block <= 256
            assert disp_warp in (8, 16)
            assert comb_warp in (8, 16)
