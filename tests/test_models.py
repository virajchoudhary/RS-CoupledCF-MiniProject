import numpy as np
import pytest
import torch
from torch import nn

from coupledcf.data import DataBundle
from coupledcf.models import ModelConfig, build_model


@pytest.fixture
def toy_data():
    user_features = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 1],
    ], dtype=np.int64)
    item_features = np.zeros((8, 18), dtype=np.float32)
    item_features[0, [0, 2]] = 1
    item_features[1, [0, 2]] = 1
    item_features[2, [1, 4, 7]] = 1
    item_features[3, [3, 5]] = 1
    item_features[4:, np.arange(4)] = 1
    observed = np.zeros((4, 8), dtype=bool)
    observed[:, :2] = True
    validation = np.array([[2], [3], [4], [5]], dtype=np.int64)
    test = np.array([[3], [4], [5], [6]], dtype=np.int64)
    return DataBundle(
        train_users=np.array([], dtype=np.int64),
        train_items=np.array([], dtype=np.int64),
        validation=validation,
        test=test,
        user_features=user_features,
        item_features=item_features,
        observed=observed,
        user_ids=np.arange(4, dtype=np.int64),
        item_ids=np.arange(8, dtype=np.int64),
    )


@pytest.mark.parametrize("variant", ("deepcf", "local", "global", "coupled_local", "coupled"))
def test_each_variant_returns_finite_logits_and_backpropagates(toy_data, variant):
    torch.manual_seed(7)
    model = build_model(toy_data, ModelConfig(variant=variant))
    users = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    items = torch.tensor([0, 2, 3, 4], dtype=torch.int64)

    logits = model(users, items)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


@pytest.mark.parametrize("variant", ("coupled_local", "coupled"))
def test_coupled_variants_receive_gradients_through_implicit_and_explicit_paths(toy_data, variant):
    torch.manual_seed(19)
    model = build_model(toy_data, ModelConfig(variant=variant))
    users = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.int64)
    items = torch.tensor([0, 2, 3, 4, 5, 6], dtype=torch.int64)

    model(users, items).sum().backward()

    assert model.user_id_embedding.weight.grad.abs().sum() > 0
    assert model.user_projection.weight.grad.abs().sum() > 0
    assert model.item_projection.weight.grad.abs().sum() > 0
    assert model.local_conv.weight.grad.abs().sum() > 0
    if variant == "coupled":
        start = model.local_conv.out_channels * (
            (model.config.side_dim - 2) // 2
        ) ** 2
        assert model.fusion[0].weight.grad[:, start:].abs().sum() > 0


def test_explicit_mlp_precedes_implicit_fusion(toy_data):
    model = build_model(toy_data, ModelConfig(variant='coupled')).eval()
    explicit_inputs, joint_inputs = [], []
    hooks = [model.fusion.register_forward_pre_hook(lambda module, args: explicit_inputs.append(args[0])),
             model.joint_fusion.register_forward_pre_hook(lambda module, args: joint_inputs.append(args[0]))]
    try:
        model(torch.tensor([0, 1]), torch.tensor([0, 1]))
    finally:
        for hook in hooks:
            hook.remove()
    assert explicit_inputs[0].shape == (2, 136)
    assert joint_inputs[0].shape == (2, 128)
    # These users and items have equal side attributes but different ID embeddings.
    torch.testing.assert_close(explicit_inputs[0][0], explicit_inputs[0][1], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(joint_inputs[0][0, :64], joint_inputs[0][1, :64], rtol=1e-5, atol=1e-7)
    assert not torch.equal(joint_inputs[0][0, 64:], joint_inputs[0][1, 64:])


@pytest.mark.parametrize("variant", ("local", "global"))
def test_explicit_only_variants_depend_on_features_not_entity_ids(toy_data, variant):
    model = build_model(toy_data, ModelConfig(variant=variant)).eval()
    assert model.user_id_embedding is None
    assert model.item_id_embedding is None
    assert not any("id_embedding" in name for name, _ in model.named_parameters())
    with torch.inference_mode():
        first = model(torch.tensor([0]), torch.tensor([0]))
        same_features = model(torch.tensor([1]), torch.tensor([1]))
    torch.testing.assert_close(first, same_features)


def test_local_coupling_supplies_every_outer_product_position(toy_data):
    model = build_model(toy_data, ModelConfig(variant="local")).eval()
    with torch.no_grad():
        for embedding in model.user_side_embeddings:
            embedding.weight.fill_(1)
        model.user_projection.weight.zero_()
        model.user_projection.bias.fill_(1)
        model.item_projection.weight.zero_()
        model.item_projection.bias.fill_(1)

    convolution_inputs = []
    hook = model.local_conv.register_forward_pre_hook(
        lambda _module, args: convolution_inputs.append(args[0].detach().clone())
    )
    try:
        model(torch.tensor([0, 2]), torch.tensor([0, 3]))
    finally:
        hook.remove()

    assert len(convolution_inputs) == 1
    coupling = convolution_inputs[0]
    assert coupling.shape == (2, 1, 8, 8)
    torch.testing.assert_close(coupling, torch.ones_like(coupling))


@pytest.mark.parametrize("variant", ("deepcf", "local", "global", "coupled_local", "coupled"))
def test_state_dict_roundtrip_preserves_deterministic_eval_scores(toy_data, variant, tmp_path):
    torch.manual_seed(23)
    original = build_model(toy_data, ModelConfig(variant=variant)).eval()
    users = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    items = torch.tensor([1, 2, 4, 6], dtype=torch.int64)
    with torch.inference_mode():
        expected = original(users, items)

    checkpoint = tmp_path / f"{variant}.pt"
    torch.save(original.state_dict(), checkpoint)
    restored = build_model(toy_data, ModelConfig(variant=variant)).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.inference_mode():
        actual = restored(users, items)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if variant in ("local", "global", "coupled_local", "coupled"):
        assert "user_features" in original.state_dict()
        assert "item_features" in original.state_dict()


def test_model_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="Unknown model variant"):
        ModelConfig(variant="unknown")
    with pytest.raises(ValueError, match="id_dim"):
        ModelConfig(id_dim=0)
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=1.0)
    with pytest.raises(ValueError, match="side_dim"):
        ModelConfig(variant="local", side_dim=5)


def test_forward_rejects_misaligned_and_non_integer_inputs(toy_data):
    model = build_model(toy_data, ModelConfig(variant="coupled")).eval()
    with pytest.raises(ValueError, match="aligned"):
        model(torch.tensor([0, 1]), torch.tensor([0]))
    with pytest.raises(TypeError, match="torch.int64"):
        model(torch.tensor([0.0]), torch.tensor([0]))
    with pytest.raises(IndexError, match="user ID"):
        model(torch.tensor([4]), torch.tensor([0]))
