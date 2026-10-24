"""Sparse weight sync: the diff producer that feeds the sparse trainer engine."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vime.backends.megatron_utils.update_weight.sparse_diff_source import (  # noqa: E402
    SparseDiffSource,
    SparseTrainerAdapter,
)


class _FakeSource:
    """Stands in for HfWeightSource: one full HF export per iteration."""

    def __init__(self, tensors):
        self.tensors = tensors
        self.draft = False

    def __iter__(self):
        yield from self.tensors.items()


class _FakeEngine:
    def __init__(self, *, is_sender=True):
        self.is_sender = is_sender
        self.sent = []

    def send_weights(self, patches):
        self.sent.append(list(patches))

    def shutdown(self):
        pass


def _export(base, changed):
    weight = base.clone()
    for index, value in changed.items():
        weight[index] = value
    return {"layer.weight": weight, "layer.bias": torch.zeros(8, dtype=torch.bfloat16)}


@pytest.mark.unit
def test_sparse_diff_source_sends_only_changed_elements():
    base = torch.arange(16, dtype=torch.bfloat16)
    source = _FakeSource(_export(base, {}))
    adapter = SparseTrainerAdapter(object(), SparseDiffSource(source), _FakeEngine())

    # The first sync has no baseline: the reference server loaded that checkpoint.
    adapter.send_weights()
    assert adapter.engine.sent[0] == []
    assert adapter.source.stats["sparse_changed_elements"] == 0

    source.tensors = _export(base, {3: 1.5, 7: -2.25})
    adapter.send_weights()
    patches = adapter.engine.sent[1]
    assert len(patches) == 1
    assert patches[0].name == "layer.weight"
    assert tuple(patches[0].full_shape) == (16,)
    assert patches[0].indices.tolist() == [3, 7]
    assert patches[0].indices.dtype == torch.int32
    assert torch.equal(patches[0].values, torch.tensor([1.5, -2.25], dtype=torch.bfloat16))

    # A round that changed nothing must not touch the wire.
    adapter.send_weights()
    assert adapter.engine.sent[2] == []
    assert adapter.source.stats["sparse_changed_elements"] == 0


@pytest.mark.unit
def test_sparse_diff_source_skips_non_sender_ranks():
    base = torch.zeros(16, dtype=torch.bfloat16)
    source = _FakeSource(_export(base, {1: 2.0}))
    diff_source = SparseDiffSource(source)
    engine = _FakeEngine(is_sender=False)

    SparseTrainerAdapter(object(), diff_source, engine).send_weights()

    # Only the sender diffs: other ranks must not hold a host snapshot.
    assert engine.sent == []
    assert diff_source.stats == {}
    assert diff_source._previous == {}


@pytest.mark.unit
def test_sparse_diff_source_rejects_draft():
    diff_source = SparseDiffSource(_FakeSource(_export(torch.zeros(8, dtype=torch.bfloat16), {})))
    with pytest.raises(RuntimeError, match="no draft"):
        diff_source.draft = True
