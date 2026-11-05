def select_device(requested="auto"):
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested in {"cuda", "mps"}:
            raise RuntimeError(f"{requested} requested, but PyTorch is not installed")
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return "cuda"
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if not (mps and mps.is_available()):
            raise RuntimeError("MPS requested but unavailable")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps and mps.is_available():
        return "mps"
    return "cpu"
