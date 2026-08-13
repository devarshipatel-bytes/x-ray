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

## Organizing your download

The default `dataset.root: data/OPIXray` is **relative to the repo root**, and `data/` is
already in `.gitignore`, so the dataset never gets committed. Three ways to wire it up:

**A. Move it in (simplest)**

```bash
cd /path/to/x-ray
unzip -q ~/Downloads/OPIXray.zip -d /tmp/opix     # unzip somewhere scratch first
ls /tmp/opix                                      # find the real top-level folder
mkdir -p data
mv /tmp/opix/<ACTUAL_TOP_FOLDER> data/OPIXray
```

**B. Symlink it** — keeps the dataset on another disk:

```bash
ln -s /mnt/d/datasets/OPIXray /path/to/x-ray/data/OPIXray
```

**C. Leave it where it is** and pass the path at runtime; every entry point takes
`--data-root`:

```bash
python -m engine.train --config configs/xdetr_opixray.yaml --data-root /mnt/d/datasets/OPIXray
```

Then verify before training — this catches wrong folder nesting and class-name mismatches
in a couple of seconds:

```bash
python -m scripts.check_dataset --config configs/xdetr_opixray.yaml
```

If your authorized copy is a direct Google Drive file (e.g. a share link/ID from the
authors), `gdown` handles Drive's large-file confirmation flow better than `wget`/`curl`:

```bash
pip install gdown
gdown --id <YOUR_OWN_FILE_ID> -O opixray.zip
unzip -q opixray.zip -d extracted   # inspect the layout before moving into data/OPIXray
```

Don't commit a specific file ID/link into this repo — it's your own authorized access,
not something to bake into version-controlled, potentially-public files.

> Scope note: use only for authorized screening-decision-support research. Do not use
> for, or frame as, guidance on concealment or defeating screening.
