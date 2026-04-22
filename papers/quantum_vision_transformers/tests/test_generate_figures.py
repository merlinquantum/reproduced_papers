from __future__ import annotations


def test_variant_key_distinguishes_family_and_profile(load_generate_figures_module) -> None:
    gf = load_generate_figures_module

    generic_full = gf.make_variant_key("A", "generic", "full")
    butterfly_lite = gf.make_variant_key("A", "butterfly", "lite")

    assert generic_full != butterfly_lite
    assert gf.pretty_model_label(generic_full) == "A: OrthoPatch [generic, full]"
    assert gf.pretty_model_label(butterfly_lite) == "A: OrthoPatch [butterfly, lite]"


def test_variant_key_supports_paper_baselines(load_generate_figures_module) -> None:
    gf = load_generate_figures_module
    variant = gf.make_variant_key("VisionTransformer", "generic", "full")
    assert gf.pretty_model_label(variant) == "VisionTransformer [baseline, full]"


def test_variant_key_distinguishes_data_regimes(load_generate_figures_module) -> None:
    gf = load_generate_figures_module
    standard = gf.make_variant_key("B", "butterfly", "lite")
    retina_sized = gf.make_variant_key("B", "butterfly", "lite", "retina_sized_train")

    assert standard != retina_sized
    assert gf.pretty_model_label(retina_sized) == "B: OrthoTransformer [butterfly, lite, retina_sized_train]"


def test_group_by_separates_generic_butterfly_and_full_sector(load_generate_figures_module) -> None:
    gf = load_generate_figures_module
    results = [
        {
            "model_type": "A",
            "dataset": "retinamnist",
            "config": {"circuit_family": "generic", "profile": "full"},
        },
        {
            "model_type": "A",
            "dataset": "retinamnist",
            "config": {"circuit_family": "butterfly", "profile": "lite"},
        },
        {
            "model_type": "D",
            "dataset": "retinamnist",
            "config": {
                "circuit_family": "generic",
                "profile": "full",
                "compound_readout": "full_sector",
            },
        },
    ]

    groups = gf.group_by(results)

    assert gf.make_variant_key("A", "generic", "full") in groups
    assert gf.make_variant_key("A", "butterfly", "lite") in groups
    assert gf.make_variant_key("D_full", "generic", "full") in groups
