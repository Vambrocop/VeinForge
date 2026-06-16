import pytest

torch = pytest.importorskip("torch")          # losses need the [dl] extra
from veinforge.losses import cl_dice_loss, dice_bce_cldice_loss


def _vertical_line(h=32, w=32):
    t = torch.zeros(1, 1, h, w)
    t[0, 0, :, w // 2 - 1:w // 2 + 2] = 1.0
    return t


def test_cldice_lower_for_matching_than_mismatching():
    t = _vertical_line()
    match_logits = t * 12 - 6                   # sigmoid ~ t
    t2 = torch.zeros_like(t)
    t2[0, 0, t.shape[2] // 2 - 1:t.shape[2] // 2 + 2, :] = 1.0   # horizontal line
    mis_logits = t2 * 12 - 6
    lm = cl_dice_loss(match_logits, t).item()
    lx = cl_dice_loss(mis_logits, t).item()
    assert lm < lx
    assert 0.0 <= lm < 0.6


def test_combined_loss_finite_and_nonnegative():
    t = _vertical_line()
    loss = dice_bce_cldice_loss(t * 12 - 6, t, cldice_weight=0.5)
    assert torch.isfinite(loss) and loss.item() >= 0.0
