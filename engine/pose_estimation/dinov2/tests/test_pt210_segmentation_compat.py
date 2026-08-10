from mmengine.config import Config, ConfigDict

from dinov2.eval.segmentation._compat import upgrade_test_pipeline


def _legacy_pipeline():
    return [
        dict(type="LoadImageFromFile"),
        dict(
            type="MultiScaleFlipAug",
            img_scale=(2048, 512),
            img_ratios=[0.5, 1.0],
            flip=True,
            transforms=[
                dict(type="Resize", keep_ratio=True),
                dict(type="RandomFlip"),
                dict(type="Normalize", mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
                dict(type="ImageToTensor", keys=["img"]),
                dict(type="Collect", keys=["img"]),
            ],
        ),
    ]


def test_upgrade_test_pipeline_accepts_plain_dict():
    cfg = upgrade_test_pipeline(dict(test_pipeline=_legacy_pipeline()))
    assert cfg["test_pipeline"][-1]["type"] == "PackSegInputs"
    assert cfg["dinov2_tta"] == dict(
        base_scale=(2048, 512), ratios=[0.5, 1.0], allow_flip=True
    )


def test_upgrade_test_pipeline_accepts_mmengine_configs():
    for cfg in (
        Config(dict(test_pipeline=_legacy_pipeline())),
        ConfigDict(test_pipeline=_legacy_pipeline()),
    ):
        upgraded = upgrade_test_pipeline(cfg)
        assert upgraded.test_pipeline[-1]["type"] == "PackSegInputs"
        assert upgraded.dinov2_tta.allow_flip is True
