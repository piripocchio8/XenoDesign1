from xenodesign.parallel import shard_round_robin


def test_shard_round_robin_balances():
    assert shard_round_robin(["a", "b", "c", "d", "e"], 2) == [["a", "c", "e"], ["b", "d"]]


def test_shard_round_robin_more_shards_than_items():
    assert shard_round_robin(["a"], 2) == [["a"], []]
