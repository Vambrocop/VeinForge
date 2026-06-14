from veinforge.params import Params


def test_params_yaml_roundtrip(tmp_path):
    p = Params(pixel_size_um=1.5, sato_sigmas=(1.0, 2.0, 3.0), min_object_px=128)
    out = tmp_path / "params.yaml"
    p.to_yaml(out)
    loaded = Params.from_yaml(out)
    assert loaded.pixel_size_um == 1.5
    assert tuple(loaded.sato_sigmas) == (1.0, 2.0, 3.0)
    assert loaded.min_object_px == 128


def test_params_defaults():
    p = Params()
    assert p.threshold_method == "otsu"
    assert p.invert is True
    assert "pixel_size_um" in p.to_dict()
