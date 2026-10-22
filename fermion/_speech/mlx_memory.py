"""MLX allocator cache helpers for long-lived Phonon-2 serve processes."""
from __future__ import annotations


def clear_mlx_cache(*, synchronize: bool = False) -> None:
    """Drop MLX's free-list cache. Safe on CPU-only installs (no-op)."""
    try:
        import mlx.core as mx

        if synchronize:
            mx.synchronize()
        mx.clear_cache()
    except Exception:
        pass


def release_automaton_mlx_tables(automaton) -> None:
    """Release device tables held on a hotword automaton."""
    if automaton is None:
        return
    if getattr(automaton, "_mlx_tables", None) is not None:
        automaton._mlx_tables = None
