# Baseline: Faster R-CNN, 5-fold ensemble

Faster R-CNN v2 (ResNet-50 FPN, COCO weights) on 512 px images, one model per fold (5 folds).

## Training

```bash
for fold in 0 1 2 3 4; do
  uv run train \
    dataset.train.fold=$fold \
    dataloader.train.batch_size=32 \
    dataloader.train.num_workers=10 \
    dataloader.val.batch_size=64 \
    dataloader.test.batch_size=64 \
    optimizer.lr=4.0e-2 \
    lr_scheduler.warmup_steps=250 \
    trainer.total_steps=6000 \
    trainer.eval_every_steps=250 \
    trainer.checkpoint_every_steps=1000 \
    trainer.ema.decay=0.998 \
    trainer.ema.warmup_steps=500 \
    trainer.compile='{fullgraph: true}'
done
```

## Submission

The predictions of the 5 fold models are fused with WBF. `--add-no-finding` adds a "No finding" prediction with confidence `1 - max box score` to every image.

```bash
uv run python scripts/make_submission.py \
  --predictions data/saved/baseline_frcnn_fold{0,1,2,3,4}/test/predictions.json \
  --add-no-finding \
  --output submission.csv
```

## Results

| Public LB | Private LB |
|---:|---:|
| 0.235 | 0.240 |
