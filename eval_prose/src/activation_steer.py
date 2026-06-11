# activation_steering.py  – v0.2
import torch
from contextlib import contextmanager
from typing import Sequence, Union, Iterable, Optional


class ActivationSteerer:
    """

    Supports restricting application to prompt tokens vs generated tokens
    via the optional prompt_length argument.
    """

    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",       # GPT‑2/Neo, Bloom, etc.
        "encoder.layer",       # BERT/RoBERTa
        "model.layers",        # Llama/Mistral
        "gpt_neox.layers",     # GPT‑NeoX
        "block",               # Flan‑T5
    )

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        positions: str = "all",
        prompt_length: Optional[int] = None,
        debug: bool = False,
    ):
        self.model, self.coeff, self.layer_idx = model, 1.0, layer_idx#float(coeff), layer_idx
        self.positions = positions.lower()
        self.prompt_length = prompt_length
        self.debug = debug
        self._handle = None

        # --- build vector ---
        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device=p.device)
        if self.vector.ndim != 1:
            raise ValueError("steering_vector must be 1‑D")
        hidden = getattr(model.config, "hidden_size", None)
        if hidden and self.vector.numel() != hidden:
            raise ValueError(
                f"Vector length {self.vector.numel()} ≠ model hidden_size {hidden}"
            )
        # Check if positions is valid
        valid_positions = {"all", "prompt", "response"}
        if self.positions not in valid_positions:
            raise ValueError("positions must be 'all', 'prompt', 'response'")

    # ---------- helpers ----------
    def _locate_layer(self):
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:  # found a full match
                if not hasattr(cur, "__getitem__"):
                    continue  # not a list/ModuleList
                if not (-len(cur) <= self.layer_idx < len(cur)):
                    raise IndexError("layer_idx out of range")
                if self.debug:
                    print(f"[ActivationSteerer] hooking {path}[{self.layer_idx}]")
                return cur[self.layer_idx]

        raise ValueError(
            "Could not find layer list on the model. "
            "Add the attribute name to _POSSIBLE_LAYER_ATTRS."
        )

    def _hook_fn(self, module, ins, out):
        steer = self.vector# self.coeff * self.vector  # (hidden,)

        def _apply_on_tokens(t: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
            if mask is None:
                return t + steer.to(t.device)
            # mask: (seq_len,) -> (1, seq_len, 1) broadcast
            mask_b = mask.to(t.device).view(1, -1, 1)
            return t + mask_b * steer.to(t.device)

        def _add(t: torch.Tensor):
            seq_len = t.shape[1]
            if self.positions == "all":
                return _apply_on_tokens(t, None)
            elif self.positions == "prompt":
                # Only affect prompt tokens (initial forward with full prompt)
                if seq_len == 1:
                    return t  # decoding step: generated token only
                if self.prompt_length is None:
                    return t  # cannot determine prompt span
                mask = torch.zeros(seq_len, dtype=t.dtype, device=t.device)
                mask[: min(self.prompt_length, seq_len)] = 1.0
                return _apply_on_tokens(t, mask)
            elif self.positions == "response":
                # Affect generated tokens only
                if seq_len == 1:
                    return t + steer.to(t.device)
                if self.prompt_length is None:
                    return t  # cannot determine response span reliably
                mask = torch.zeros(seq_len, dtype=t.dtype, device=t.device)
                mask[min(self.prompt_length, seq_len) : seq_len] = 1.0
                return _apply_on_tokens(t, mask)
            else:
                raise ValueError(f"Invalid positions: {self.positions}")

        # out may be tensor or tuple/list => normalise to tuple
        if torch.is_tensor(out):
            new_out = _add(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                # unusual case – don't touch
                return out
            head = _add(out[0])
            new_out = (head, *out[1:])  # keep other entries
        else:
            return out  # unknown type – leave unchanged

        if self.debug:
            with torch.no_grad():
                delta = (new_out[0] if isinstance(new_out, tuple) else new_out) - (
                    out[0] if isinstance(out, (tuple, list)) else out
                )
                print(
                    "[ActivationSteerer] |delta| (mean ± std): "
                    f"{delta.abs().mean():.4g} ± {delta.std():.4g}"
                )
        return new_out

    # ---------- context manager ----------
    def __enter__(self):
        layer = self._locate_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        self.remove()  # always clean up

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


class ActivationSteererMultiple:
    """
    Add multiple (coeff * steering_vector) to chosen transformer block outputs.
    Accepts a list of dicts, each with keys: steering_vector, coeff, layer_idx, positions.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        instructions: Sequence[dict],
        *,
        debug: bool = False,
    ):
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._handles = []
        self._steerers = []

        # Validate and create individual steerers
        for inst in self.instructions:
            steerer = ActivationSteerer(
                model,
                inst["steering_vector"],
                coeff=inst.get("coeff", 0.0),
                layer_idx=inst.get("layer_idx", -1),
                positions=inst.get("positions", "all"),
                prompt_length=inst.get("prompt_length", None),
                debug=debug,
            )
            self._steerers.append(steerer)

    def __enter__(self):
        for steerer in self._steerers:
            layer = steerer._locate_layer()
            handle = layer.register_forward_hook(steerer._hook_fn)
            steerer._handle = handle
            self._handles.append(handle)
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        for steerer in self._steerers:
            steerer.remove()
        self._handles.clear()