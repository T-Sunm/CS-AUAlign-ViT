import pytest
import torch
from torch.nn import functional as F

from cs_aualign_vit.models import AdversarialAlignment, DomainDiscriminator
from cs_aualign_vit.models.post.adversarial import domain_accuracy, grad_reverse


def make(dim=4, **kwargs):
    kwargs.setdefault("dropout", 0.0)
    torch.manual_seed(0)
    return AdversarialAlignment(input_dim=dim, hidden_dim=8, **kwargs)


@pytest.mark.parametrize("batch", [1, 3])
def test_logits_have_batch_shape(batch):
    assert DomainDiscriminator(4, 8, 0.0)(torch.randn(batch, 4)).shape == (batch,)


def test_loss_is_sum_of_separate_bce_means_with_unequal_batches():
    module = make()
    source, target = torch.randn(5, 4), torch.randn(2, 4)
    loss = module(source, target)
    z_s, z_t = module.discriminator(source), module.discriminator(target)
    expected = F.binary_cross_entropy_with_logits(z_s, torch.ones(5))
    expected = expected + F.binary_cross_entropy_with_logits(z_t, torch.zeros(2))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_grl_reverses_feature_gradient_but_not_discriminator_gradient():
    source, target = torch.randn(3, 4), torch.randn(3, 4)
    results = []
    for use_grl in (True, False):
        module = make().eval()
        s, t = source.clone().requires_grad_(), target.clone().requires_grad_()
        if use_grl:
            module(s, t).backward()
        else:
            z_s, z_t = module.discriminator(s), module.discriminator(t)
            (
                F.binary_cross_entropy_with_logits(z_s, torch.ones(3))
                + F.binary_cross_entropy_with_logits(z_t, torch.zeros(3))
            ).backward()
        results.append((s.grad, t.grad, module.discriminator.net[0].weight.grad))
    assert torch.allclose(results[0][0], -results[1][0], atol=1e-6)
    assert torch.allclose(results[0][1], -results[1][1], atol=1e-6)
    assert torch.allclose(results[0][2], results[1][2], atol=1e-6)


def test_lambda_scales_reversed_gradient():
    x = torch.ones(2, 3, requires_grad=True)
    grad_reverse(x, 0.5).sum().backward()
    assert torch.equal(x.grad, torch.full((2, 3), -0.5))


def test_detached_features_still_train_discriminator():
    module = make()
    module(torch.randn(2, 4).detach(), torch.randn(2, 4).detach()).backward()
    assert all(p.grad is not None for p in module.parameters())


def test_config_round_trip_and_weights_only_loadable(tmp_path):
    module = make(lambda_grl=0.7)
    path = tmp_path / "config.pt"
    torch.save(module.config(), path)
    config = torch.load(path, weights_only=True)
    assert AdversarialAlignment.from_config(config).config() == module.config()


def test_domain_accuracy_on_known_logits():
    metrics = domain_accuracy(torch.tensor([1.0, -1.0, 2.0, 3.0]), torch.tensor([-1.0, 1.0]))
    assert metrics == {"acc_s": 0.75, "acc_t": 0.5, "acc_balanced": 0.625}


def test_forward_stores_metrics_and_validates_shapes():
    module = make()
    module(torch.randn(2, 4), torch.randn(3, 4))
    assert set(module.last_metrics) == {"acc_s", "acc_t", "acc_balanced"}
    for bad in (torch.randn(4), torch.randn(0, 4), torch.randn(2, 5)):
        with pytest.raises(ValueError):
            module(bad, torch.randn(2, 4))
