# Getting OPIXray

OPIXray is released by the authors (Wei et al., ACM MM 2020) and requires a signed
academic agreement — it is **not** downloadable anonymously. Steps:

1. Go to the official repo: https://github.com/OPIXray-author/OPIXray
2. Follow their instructions to request access (sign + email the agreement form).
3. You receive a Google Drive link. Download and unzip it.

## Expected layout (what this project's loader assumes)

```
data/OPIXray/
  train/
    train_image/*.jpg
    train_annotation/*.txt          # one .txt per image, lines: <name> <class> xmin ymin xmax ymax
  test/
    test_image/*.jpg
    test_annotation/*.txt
    test_occlusion/
      OL1.txt   OL2.txt   OL3.txt   # image stems per occlusion level
```

The loader (`data/opixray.py`) is tolerant of a few folder-name variants and of the
`<class> xmin ymin xmax ymax` annotation form. If your copy differs:

* set `dataset.root` in `configs/xdetr_opixray.yaml` to your unzip path;
* if annotations are Pascal-VOC XML, add `ann_format: voc_xml` under `dataset:`.

## Classes (order matters — must match the config)

```
Folding_Knife, Straight_Knife, Scissor, Utility_Knife, Multi-tool_Knife
```

## On Colab

Put the unzipped `OPIXray/` folder in your Google Drive (e.g. `MyDrive/xray/OPIXray`)
and point `dataset.root` there so it survives session restarts. See
`notebooks/colab_train.ipynb`.

> Scope note: use only for authorized screening-decision-support research. Do not use
> for, or frame as, guidance on concealment or defeating screening.
