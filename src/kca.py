"""Shared utilities for the KCA diagnostic across datasets."""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def entropy_nats(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Shannon entropy in nats. logits: [..., V]."""
    logp = F.log_softmax(logits / temperature, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def jsd_nats(logits_a: torch.Tensor, logits_b: torch.Tensor, temperature: float) -> torch.Tensor:
    """Jensen-Shannon divergence in nats. Inputs: [..., V]."""
    logp_a = F.log_softmax(logits_a / temperature, dim=-1)
    logp_b = F.log_softmax(logits_b / temperature, dim=-1)
    p_a = logp_a.exp()
    p_b = logp_b.exp()
    logm = torch.logaddexp(logp_a, logp_b) - torch.log(
        torch.tensor(2.0, device=logits_a.device, dtype=logits_a.dtype)
    )
    kl_a = (p_a * (logp_a - logm)).sum(dim=-1)
    kl_b = (p_b * (logp_b - logm)).sum(dim=-1)
    return 0.5 * (kl_a + kl_b)


class LayerCaptures:
    """Forward-hook container capturing a^l and m^l for every decoder layer."""

    def __init__(self, model):
        self.n_layers = len(model.model.layers)
        self.a: List[Optional[torch.Tensor]] = [None] * self.n_layers
        self.m: List[Optional[torch.Tensor]] = [None] * self.n_layers
        self._handles = []

        for i, layer in enumerate(model.model.layers):
            self._handles.append(
                layer.self_attn.register_forward_hook(self._make_attn_hook(i))
            )
            self._handles.append(
                layer.mlp.register_forward_hook(self._make_mlp_hook(i))
            )

    def _make_attn_hook(self, i: int):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.a[i] = t
        return hook

    def _make_mlp_hook(self, i: int):
        def hook(_mod, _inp, out):
            self.m[i] = out
        return hook

    def reset(self):
        self.a = [None] * self.n_layers
        self.m = [None] * self.n_layers

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def compute_kca_features(
    model,
    input_ids: torch.Tensor,
    response_start: int,
    captures: LayerCaptures,
    temperature: float,
    lens_dtype: torch.dtype,
    lens_weight: torch.Tensor,
    lens_bias: Optional[torch.Tensor],
    emit_dola: bool = False,
) -> Dict[str, List[List[float]]]:
    """One teacher-forced forward pass, then per-layer logit lens over response tokens.

    Always returns L x T python lists for: H_a, H_m, H_x, H_pre, H_p, JSD_am.

    Here:
        H_pre -- entropy of the intermediate state x^{ell-1} + a^ell
        H_p   -- entropy of the additive path state a^ell + m^ell

    When ``emit_dola`` is True, additionally returns:
        JSD_to_final -- JSD(lens(x^ell), lens(x^L)) for each ell in [0, L-1].
        This is the DoLa-style (Chuang et al., ICLR 2024) "layer contrast"
        signal, repurposed as a hallucination-detection feature for our
        head-to-head comparison.

    Lens projection is done in ``lens_dtype`` using a pre-cast copy of
    ``lm_head.weight`` to avoid fp16 overflow at the final decoder layer
    (Qwen2.5 residual stream can exceed fp16 range at layer L-1).
    """
    captures.reset()
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states  # (embed, layer1_out, ..., layerL_out)

    norm = model.model.norm
    L = captures.n_layers
    H_a: List[List[float]] = []
    H_m: List[List[float]] = []
    H_x: List[List[float]] = []
    H_pre: List[List[float]] = []
    H_p: List[List[float]] = []
    JSD_am: List[List[float]] = []
    JSD_to_final: Optional[List[List[float]]] = [] if emit_dola else None

    def lens(t: torch.Tensor) -> torch.Tensor:
        normed = norm(t).to(lens_dtype)
        return F.linear(normed, lens_weight, lens_bias)

    # Pre-compute final-layer logit-lens distribution once if DoLa is requested.
    if emit_dola:
        x_L = hidden_states[L][0, response_start:]
        logits_L = lens(x_L)
        logp_L = F.log_softmax(logits_L / temperature, dim=-1)
        p_L = logp_L.exp()
        log2 = torch.log(
            torch.tensor(2.0, device=logp_L.device, dtype=logp_L.dtype)
        )

    for l in range(L):
        a_l = captures.a[l][0, response_start:]
        m_l = captures.m[l][0, response_start:]
        x_l = hidden_states[l + 1][0, response_start:]
        x_prev = hidden_states[l][0, response_start:]
        pre_l = x_prev + a_l
        p_l = a_l + m_l

        logits_a = lens(a_l)
        logits_m = lens(m_l)
        logits_x = lens(x_l)
        logits_pre = lens(pre_l)
        logits_p = lens(p_l)

        H_a.append(entropy_nats(logits_a, temperature).tolist())
        H_m.append(entropy_nats(logits_m, temperature).tolist())
        H_x.append(entropy_nats(logits_x, temperature).tolist())
        H_pre.append(entropy_nats(logits_pre, temperature).tolist())
        H_p.append(entropy_nats(logits_p, temperature).tolist())
        JSD_am.append(jsd_nats(logits_a, logits_m, temperature).tolist())

        if emit_dola:
            logp_l = F.log_softmax(logits_x / temperature, dim=-1)
            p_l = logp_l.exp()
            logm = torch.logaddexp(logp_l, logp_L) - log2
            kl_l = (p_l * (logp_l - logm)).sum(dim=-1)
            kl_L = (p_L * (logp_L - logm)).sum(dim=-1)
            JSD_to_final.append((0.5 * (kl_l + kl_L)).tolist())
            del logp_l, p_l, logm, kl_l, kl_L

        del logits_a, logits_m, logits_x, logits_pre, logits_p

    out = {
        "H_a": H_a,
        "H_m": H_m,
        "H_x": H_x,
        "H_pre": H_pre,
        "H_p": H_p,
        "JSD_am": JSD_am,
    }
    if emit_dola:
        out["JSD_to_final"] = JSD_to_final
    return out
