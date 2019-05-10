## Network Architecture
Using [netron](https://github.com/lutzroeder/netron) tool to visualize, see the [result](./cfg/yolov3-voc.png). The file to visualize locates in `DARKNET_ROOT/cfg/yolov3-voc.cfg`. Another useful light tool is [NN-SVG](http://alexlenail.me/NN-SVG/LeNet.html).
```
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32  0.299 BFLOPs
    1 conv     64  3 x 3 / 2   416 x 416 x  32   ->   208 x 208 x  64  1.595 BFLOPs
    2 conv     32  1 x 1 / 1   208 x 208 x  64   ->   208 x 208 x  32  0.177 BFLOPs
    3 conv     64  3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  64  1.595 BFLOPs
    4 res    1                 208 x 208 x  64   ->   208 x 208 x  64
    5 conv    128  3 x 3 / 2   208 x 208 x  64   ->   104 x 104 x 128  1.595 BFLOPs
    6 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64  0.177 BFLOPs
    7 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128  1.595 BFLOPs
    8 res    5                 104 x 104 x 128   ->   104 x 104 x 128
    9 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64  0.177 BFLOPs
   10 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128  1.595 BFLOPs
   11 res    8                 104 x 104 x 128   ->   104 x 104 x 128
   12 conv    256  3 x 3 / 2   104 x 104 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   13 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   14 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   15 res   12                  52 x  52 x 256   ->    52 x  52 x 256
   16 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   17 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   18 res   15                  52 x  52 x 256   ->    52 x  52 x 256
   19 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   20 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   21 res   18                  52 x  52 x 256   ->    52 x  52 x 256
   22 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   23 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   24 res   21                  52 x  52 x 256   ->    52 x  52 x 256
   25 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   26 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   27 res   24                  52 x  52 x 256   ->    52 x  52 x 256
   28 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   29 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   30 res   27                  52 x  52 x 256   ->    52 x  52 x 256
   31 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   32 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   33 res   30                  52 x  52 x 256   ->    52 x  52 x 256
   34 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
   35 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
   36 res   33                  52 x  52 x 256   ->    52 x  52 x 256
   37 conv    512  3 x 3 / 2    52 x  52 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   38 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   39 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   40 res   37                  26 x  26 x 512   ->    26 x  26 x 512
   41 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   42 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   43 res   40                  26 x  26 x 512   ->    26 x  26 x 512
   44 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   45 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   46 res   43                  26 x  26 x 512   ->    26 x  26 x 512
   47 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   48 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   49 res   46                  26 x  26 x 512   ->    26 x  26 x 512
   50 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   51 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   52 res   49                  26 x  26 x 512   ->    26 x  26 x 512
   53 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   54 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   55 res   52                  26 x  26 x 512   ->    26 x  26 x 512
   56 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   57 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   58 res   55                  26 x  26 x 512   ->    26 x  26 x 512
   59 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   60 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   61 res   58                  26 x  26 x 512   ->    26 x  26 x 512
   62 conv   1024  3 x 3 / 2    26 x  26 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   63 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   64 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   65 res   62                  13 x  13 x1024   ->    13 x  13 x1024
   66 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   67 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   68 res   65                  13 x  13 x1024   ->    13 x  13 x1024
   69 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   70 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   71 res   68                  13 x  13 x1024   ->    13 x  13 x1024
   72 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   73 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   74 res   71                  13 x  13 x1024   ->    13 x  13 x1024
   75 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   76 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   77 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   78 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   79 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512  0.177 BFLOPs
   80 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   81 conv     75  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x  75  0.026 BFLOPs
   82 yolo
   83 route  79
   84 conv    256  1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x 256  0.044 BFLOPs
   85 upsample            2x    13 x  13 x 256   ->    26 x  26 x 256
   86 route  85 61
   87 conv    256  1 x 1 / 1    26 x  26 x 768   ->    26 x  26 x 256  0.266 BFLOPs
   88 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   89 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   90 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   91 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256  0.177 BFLOPs
   92 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512  1.595 BFLOPs
   93 conv     75  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x  75  0.052 BFLOPs
   94 yolo
   95 route  91
   96 conv    128  1 x 1 / 1    26 x  26 x 256   ->    26 x  26 x 128  0.044 BFLOPs
   97 upsample            2x    26 x  26 x 128   ->    52 x  52 x 128
   98 route  97 36
   99 conv    128  1 x 1 / 1    52 x  52 x 384   ->    52 x  52 x 128  0.266 BFLOPs
  100 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
  101 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
  102 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
  103 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128  0.177 BFLOPs
  104 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
  105 conv     75  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x  75  0.104 BFLOPs
  106 yolo
```
Notice: there are total 3 yolo layers, `layer 82`, `layer 94` and `layer 106`. The input feature map of yolo layers are `13 x 13 x 75`, ` 26 x 26 x 75` and `52 x 52 x 75`, at the same time, smaller feature map means bigger receptive field and vice versa. So the detection granularity of objects is becoming more and more fine. The size of input image is `416 x 416`, as the convolved input feature map of yolo layers are `13 x 13 x 75`, ` 26 x 26 x 75` and `52 x 52 x 75`, meaning we apply `416/13=32X`, 16X, and 8X downsample, respectively. We defined 9 prior anchors with size of `anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326`(please check `./cfg/yolov3-voc.cfg` file), which means the anchor has size of 10 pixel x 13 pixel, et al. When execute detection, the center point of anchors is moved pixel by pixel in input feature map. So what kind of input feature map will use which type of anchors? Here is a table:  

| size of feature map | anchors|
| ------------------- | -------|
| 13 x 13 x 75        |(116,90),(156,198),(373,326)|
| 26 x 26 x 75        |(30,61),(62,45),(59,119)|
| 52 x 52 x 75        |(10,13),(16,30),(33,23)|

You can image that, as the feature map is bigger and bigger, the size of anchors are smaller and smaller, which means that the anchors detect more smaller objects and move stride becomes small (Why? At first, 416/13(feature map size)=32 (piexel)/(a pixel of anchor), if we move a pixel in anchor, meaning we 'see' 32 pixels in input image. If 8X downsample, meaning we only 'see' 8 pixels when we move a step of anchor in the input feature map.)

Why the dept of input feature map of yolo layer is 75? Because in VOC dataset, we have 20 classes, for each class, we want to predict a probability for it, for the maximum probability of a object, we predict a box for that object. For a box, we predict a confidence for it. So we have 20 (classes) + 4 (x, y, w, h of box) + 1 (confidence of box) = 25. There are 3 anchors for a yolo layer, so 25 x 3 = 75.

## VOC Data Preprocess
Follow the [tutorial](https://pjreddie.com/darknet/yolo/), we need train lists and labels. After run `DARKNET_ROOT/scripts/voc_label.py`, we get train, test, and val TXT files, and the label file is in `VOCdevkit/VOC2007/labels`, and `VOCdevkit/VOC2012/labels`. For each file in labels folder, such as `2008_000015.txt`, coming from `2008_000015.xml`. The detials in `2008_000015.txt`:  
```
4 0.646 0.26758409785932724 0.216 0.5351681957186545
4 0.219 0.22782874617737003 0.214 0.45565749235474007
```
Here 4 is 5-th in `voc.names` file, means the class is bottle.
(0.646, 0.26758409785932724) is the scaled center point of obj 1.   
0.646 = ((378-270)/2 + 270 - 1) / 500 ( Why here mimus 1? Maybe is trivial)  
0.26758409785932724 = ((176-1)/2 + 1 -1) / 327  
(0.216, 0.5351681957186545) is the scaled width and height of obj 1.  
0.216 = (378-270) / 500
0.5351681957186545 = (176-1) / 327
```
|--------500--------------|
| (270,1)                 |
|    ---------            |
|    | obj 1 |            327
|    ---------(378,176)   |
|                         |
|                         |
|-------------------------|
```

A piece of `DARKNET_ROOT/data/voc.names`:
```
  1 aeroplane
  2 bicycle
  3 bird
  4 boat
  5 bottle
...
```
A piece of `2008_000015.xml`:
```
...
    <size>
        <width>500</width>
        <height>327</height>
        <depth>3</depth>
    </size>
    <segmented>1</segmented>
    <object>
        <name>bottle</name>
        <pose>Unspecified</pose>
        <truncated>1</truncated>
        <occluded>1</occluded>
        <bndbox>
            <xmin>270</xmin>
            <ymin>1</ymin>
            <xmax>378</xmax>
            <ymax>176</ymax>
        </bndbox>
        <difficult>0</difficult>
    </object>
    <object>
        <name>bottle</name>
        <pose>Unspecified</pose>
        <truncated>1</truncated>
        <occluded>1</occluded>
        <bndbox>
            <xmin>57</xmin>
            <ymin>1</ymin>
            <xmax>164</xmax>
            <ymax>150</ymax>
        </bndbox>
...
```

## Details of yolo layer
In theory, the input size of yolo layer ( layer 82) is 13 x 13. However, if set `jitter` and `random=1` in `yolov3_voc.cfg` file, means for a new batch images to be trained, we randomly resize image and resize network at the same time. So the input to yolo layer (layer 82) may not be 13 x 13. For an example, if the input resized image is 512 x 512, the input size of layer 82 (yolo layer) is 16 x 16. More details please check `detector.c:79:resize_network()`.

Okey, now let's go deep in yolo layer. See [yolo_layer.c](./src/yolo_layer.c) and `box.c`. If you want to re-train the pre-trained weights, set `max_batches` bigger than `get_current_batch(num) == 32013312`. Otherwise, the procedure just saves `yolov3.weights` (237M) to `backup` folder.

```
$ gdb ./darknet
(gdb) b detector.c:62
(gdb) run detector train cfg/voc.data cfg/yolov3-voc.cfg /data/yolov3.weights
(gdb) set net-\>max\_batches=33333333
(gdb) b forward\_yolo\_layer
(gdb) c
```

To quickly debug and only check what is going on the yolo layer, you can also use `yolov3-tiny.cfg`, and down load `yolov3-tiny.weights` (33.79M)
> wget https://pjreddie.com/media/files/yolov3-tiny.weights

```
$ gdb ./darknet
(gdb) b detector.c:62
(gdb) run detector train cfg/voc.data cfg/yolov3-tiny.cfg /data/yolov3-tiny.weights
(gdb) set net-\>max\_batches=40001111
(gdb) b forward\_yolo\_layer
(gdb) c
```

## Yolo-tiny Network Architecture

```
layer     filters    size              input                output

1 max               2 x 2 / 2   416 x 416 x  16   ->   208 x 208 x  16
2 conv      32      3 x 3 / 1   208 x 208 x  16   ->   208 x 208 x  32  0.399 BFLOPs
3 max               2 x 2 / 2   208 x 208 x  32   ->   104 x 104 x  32
4 conv      64      3 x 3 / 1   104 x 104 x  32   ->   104 x 104 x  64  0.399 BFLOPs
5 max               2 x 2 / 2   104 x 104 x  64   ->    52 x  52 x  64
6 conv     128      3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x 128  0.399 BFLOPs
7 max               2 x 2 / 2    52 x  52 x 128   ->    26 x  26 x 128
8 conv     256      3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 256  0.399 BFLOPs
9 max               2 x 2 / 2    26 x  26 x 256   ->    13 x  13 x 256
10 conv    512      3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512  0.399 BFLOPs
11 max              2 x 2 / 1    13 x  13 x 512   ->    13 x  13 x 512
12 conv   1024      3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
13 conv    256      1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 256  0.089 BFLOPs
14 conv    512      3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512  0.399 BFLOPs
15 conv    255      1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x 255  0.044 BFLOPs
16 yolo
17 route  13
18 conv    128  1 x 1 / 1    13 x  13 x 256   ->    13 x  13 x 128  0.011 BFLOPs
```

## Evaluate mAP
If you want to run Yolo-Tiny network, modify `names` in `./cfg/voc.data` to `names = data/coco.names`

```
$ vim Makefile
  set DEBUG=0
$ make all -j8
$ mkdir -p DARKNET\_ROOT/results
$ ./darknet detector valid cfg/voc.data cfg/yolov3-tiny.cfg data/yolov3-tiny.weights results\_voc.txt
```
What is mAP(mean Average Precision) just mean of AP of each classes, mAP(class-A, clasa-B) = [AP(class-A) + AP(class-B)] / 2. And AP is the area unger the precision-recall curve. What is Precision and Recall ? Precision-Recall curve just a curve of points (precision, recall).

Precision = true positive / ( true positive + false negative)  
Recall = tp / (tp + fn)  

x% precision of class-A just meaning that x% of the retrieved results were class-A, and x% recall of class-A just meaning that x% of the class-A were retrieved. More details see the script [compute_map.py](./compute_map.py).

To compute mAP of cat class, run:    
```
$ python2 compute\_AP.py
```
