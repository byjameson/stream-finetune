"""How wrong was the pre-fix masked gradient? Realistic SFT row shapes."""
import torch
import torch.nn.functional as F

from streamft.stream_quant_lora import chunked_lm_loss_and_backward_masked


def old_masked(head, h, ys, chunk=128, ignore_index=-100):
    """The pre-25.10 implementation: divide each chunk by its own valid count."""
    B, T, _ = h.shape
    total = torch.zeros((), device=h.device, dtype=torch.float32)
    valid_total = 0
    for a in range(0, T, chunk):
        b = min(a + chunk, T)
        logits = head(h[:, a:b])
        tgt = ys[:, a:b].reshape(-1)
        valid = int((tgt != ignore_index).sum())
        if valid == 0:
            continue
        loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt,
                                   reduction="sum", ignore_index=ignore_index)
        (loss_sum / valid).backward()
        with torch.no_grad():
            total = total + loss_sum.detach()
        valid_total += valid
    denom = max(valid_total, 1)
    return total / denom


for T, prompt_frac, chunk in [(512, 0.60, 128), (512, 0.85, 128), (1024, 0.60, 128),
                              (512, 0.60, 256), (128, 0.60, 64)]:
    torch.manual_seed(0)
    d, V = 64, 128
    head = torch.nn.Linear(d, V, bias=False)
    h = torch.randn(1, T, d, requires_grad=True)
    ys = torch.randint(0, V, (1, T))
    k = int(T * prompt_frac)
    ys[0, :k] = -100

    g_ref = torch.autograd.grad(
        F.cross_entropy(head(h).reshape(-1, V), ys.reshape(-1),
                        ignore_index=-100, reduction="mean"), h)[0].clone()

    h_old = h.detach().clone().requires_grad_(True)
    old_masked(head, h_old, ys, chunk=chunk)
    g_old = h_old.grad.clone()

    h_new = h.detach().clone().requires_grad_(True)
    chunked_lm_loss_and_backward_masked(head, h_new, ys, chunk=chunk)
    g_new = h_new.grad.clone()

    def rel(a, b):
        return (a - b).norm().item() / max(b.norm().item(), 1e-12)

    print(f"T={T:4d} masked={prompt_frac:.0%} chunk={chunk:3d} | "
          f"old rel_err={rel(g_old, g_ref):.3%} | new rel_err={rel(g_new, g_ref):.2e} | "
          f"cos_old={torch.nn.functional.cosine_similarity(g_old.flatten(), g_ref.flatten(), dim=0):.5f}")
