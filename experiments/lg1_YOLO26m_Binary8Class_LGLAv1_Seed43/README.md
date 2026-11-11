# lg1 formal experiment container

`EXPERIMENT=lg1`

`STRUCTURE=LGLA_V1`

`SEED=43`

`FORMAL_LONG_TRAINING_STARTED=FALSE`

The formal runner will create `results/lg1_yolo26m_lgla_v1_seed43` only when the
user launches `tools/lg1/train_lg1_from_pycharm.py` from PyCharm. Preflight code
must not create that formal result directory. Test, external, and PAD-UFES-20
data are forbidden during lg1 development.
