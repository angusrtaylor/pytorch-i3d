Setup environment
```
conda env create -f environment.yaml
conda activate pytorchi3d
```

Train RGB model
```
python train.py --cfg config/train_rgb.yaml
```

Train flow model
```
python train.py --cfg config/train_flow.yaml
```

Evaluate combined model
```
python test.py
```