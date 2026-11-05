from src.data.loaders import load_dataset
from src.data.splits import make_splits

def test_split_sizes_and_disjoint_indices():
    frame, target = load_dataset({"dataset":"synthetic_binary", "seed":42})
    cfg = {"seed":42, "split":{"strategy":"stratified", "validation_fraction":0.2, "test_fraction":0.2}}
    Xtr, Xv, Xte, *_ = make_splits(frame, target, cfg)
    assert len(Xtr) == 1500 and len(Xv) == 500 and len(Xte) == 500
    assert set(Xtr.index).isdisjoint(Xv.index) and set(Xtr.index).isdisjoint(Xte.index)
