# Age and Gender estimation based on CNN and TensorFlow

[![N|Solid](https://res.cloudinary.com/dlvaangxn/image/upload/c_scale,w_150/v1563630297/unx-logo.png)](https://www.unxdigital.com/)

![CircleCI](https://circleci.com/gh/google/wikiloop-battlefield/tree/master.svg?style=svg)

Implementing CNN to estimate age and gender from faces

![multi-face-result](https://raw.githubusercontent.com/unx-digital/Age-Gender-CNN-TensorFlow/master/images/multiple-faces-result.png)

## DEPENDENCIES
This project has following dependencies and tested under OSX with Python2.7.14. We use [virtualenv](https://virtualenv.pypa.io/en/latest/) to play safely with multiple Python versions.

- tensorflow==1.4
- dlib==19.7.99
- cv2
- matplotlib==2.1.0
- imutils==0.4.3
- numpy==1.13.3
- pandas==0.20.3


## HOW TO USE?

### One picture demo
In order to test the model in one sigle picture, run
```
python eval.py --I "./images/roger.png" --M "./models/" --font_scale 1 --thickness 1
```
![roger-result](https://raw.githubusercontent.com/unx-digital/Age-Gender-CNN-TensorFlow/master/images/roger-result.png)

Flag **--I** tells where your picture is.If the text label too small or too large on the picture,you can use a different **--font_scale 1** and **--thickness 1** to adjust the text size and thickness.


### Real time estimation using webcam
![web cam](https://raw.githubusercontent.com/unx-digital/Age-Gender-CNN-TensorFlow/master/images/webcam-result.gif)

## TRAIN YOUR OWN MODELS

### Make tfrecords
In order to train your own models,you should first download [imdb](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar) or [wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar) dataset,and then extract it under **data** path,after that,images path should look like
> /path/to/project/data/imdb_crop/00/somepictures  
/path/to/project/data/imdb_crop/01/somepictures  
...  
/path/to/project/data/imdb_crop/99/somepictures

Then you can run 
```bash
python convert_to_records_multiCPU.py --imdb --nworks 8
```
to convert images to tfrecords.**--imdb** means using imdb dataset,**--nworks 8** means using 8 cpu cores to convert the dataset parallelly.Because we will first detect and align faces in the pictures,which is a time consuming step,so we recommend to use as many cores as possible.Intel E5-2667 v4 and with 32 cores need approximately 50 minutes.

### Train model
Once you have converted images to tfrecords,you should have the following path:
> /path/to/project/data/train/train-000.tfrecords  
...  
/path/to/project/data/test/test-000.tfrecords
 
 At present, our deep CNN uses FaceNet architecture,which based on inception-resnet-v1 to extract features.To speed up training,we use the pretrained model's weight from [this project](https://github.com/davidsandberg/facenet) and have converted the weight to adapt our model, you can download this converted pretrained facenet weight checkpoint from [here](https://mega.nz/#!4G4yxbAL!D9QG48yzCeFegCFhZfpCgOyLYbfDdU6lt2k2kK9n23g) or [here](https://pan.baidu.com/s/1dFewgqH).Extract it to path **models**.
 > /path/to/project/models/checkpoint  
 /path/to/project/models/model.ckpt-0.data-00000-of-00001  
 /path/to/project/models/model.ckpt-0.index  
 /path/to/project/models/model.ckpt-0.meta
 
 **NOTE:** This step is optional,you can also train your model from scratch.
 To start training,run
 
```bash
python train.py --lr 1e-3 --weight_decay 1e-5 --epoch 6 --batch_size 128 --keep_prob 0.8 --cuda
```
**NOTE:** Using the flag **--cuda** will train the model with GPU.

Using tensorboard to visualize learning
```
tensorboard --logdir=./train_log
```

![train log](https://raw.githubusercontent.com/unx-digital/Age-Gender-CNN-TensorFlow/master/images/train_log.jpg)


### Test your model
You can test all your trained models on testset through
```
python test.py --images "./data/test" --model_path "./models" --batch_size 128 --choose_best --cuda
```
Flag **--cuda** means using GPU when testing.**--choose_best** means testing all trained models and return the best one.If you just want to test the latest saved model,without this flag.
```
python test.py --images "./data/test" --model_path "./models" --batch_size 128 --cuda
```


```
MIT License

Copyright (c) [2019] [UNX Digital]
```
