How to train?
===============

 **1. Commands**
------------ 

```
python train.py {program arguments}
```

**2. Program arguments**
-------------

| Option | Mandatory |                   Purpose                   |
|--------|-----------|:-------------------------------------------:|
|   -o   |     O     | Directory path to save checkpoints.         |
|   -c   |     X     | Path of pretrained checkpoint to load.      |
|   -l   |     O     | Log directory to drop logs for tensorboard. |

**3. Pretrained models**
-----------------------
| Model              | Pretrained checkpoint | Matching hyperparameters |
|--------------------|-----------------------|:------------------------:|
|   Soft pitchtron   |[Soft pitchtron](https://www.dropbox.com/s/z2y0ts8luo288tt/checkpoint_soft_pitchtron?dl=1)|[configs](https://www.dropbox.com/s/z2y0ts8luo288tt/checkpoint_soft_pitchtron?dl=1)                          |
|   Hard pitchtron   |[Hard pitchtron](https://www.dropbox.com/s/fsu84dprmire76s/checkpoint_hard_pitchtron?dl=1)|[configs](https://www.dropbox.com/s/tsr5ib4a1lyzggq/config_hard_pitchtron.py?dl=1)                          |
| Global style token |[GST](https://www.dropbox.com/s/3okwrwrytyx2bcx/checkpoint_gst?dl=1)|[configs](https://www.dropbox.com/s/ub81eq7aq8esx53/config_gst.py?dl=1)                          |
| WaveGlow vocoder   |[WaveGlow](https://drive.google.com/file/d/1Rm5rV5XaWWiUbIpg5385l5sh68z2bVOE/view)                       |       -                   |

