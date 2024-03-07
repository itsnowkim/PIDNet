## Train

```python
python tools/train.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml GPUS (0,1) TRAIN.BATCH_SIZE_PER_GPU 6
python tools/train.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml GPUS (0) TRAIN.BATCH_SIZE_PER_GPU 6
```

## Test

- Custom inputs 실행 (samples)

```python
python tools/custom.py --a pidnet-s --p ./pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt --r ./samples/ --t .png
```

- Evaluation

```python
python tools/eval.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml TEST.MODEL_FILE pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt
python tools/eval.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml TEST.MODEL_FILE pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt
```

## Speed Measurement

```python
python models/speed/pidnet_speed.py --a 'pidnet-s' --c 19 --r 1024 2048
```