# Play with SSD
## Tools and Materials preparation
1. Download [SSD Caffe](https://github.com/weiliu89/caffe/tree/ssd) and make
```
$ git clone https://github.com/weiliu89/caffe.git
$ cd caffe
$ git checkout ssd
$ mkdir -p build
$ cd build
$ cmake ..
$ make all -j8
```
2. Download [pre-trained model](https://drive.google.com/file/d/0BzKzrI_SkD1_WVVTSmQxU0dVRzA/view),
and extract files to anywhere you want. For model inference state, there must contanins two files: `deploy.prototxt`,` VGG_VOC0712_SSD_300x300_iter_120000.caffemodel`.  
3. generate your images or videos samples file list
e.g., I append a `dog.jpg` path to `test.txt`:  
```
/home/lkj/deeplearning/tmp_caffe/caffe/build/examples/ssd/dog.jpg
```
4. Run the SSD
```
lkj@lkj:~/deeplearning/tmp_caffe/caffe$ ./build/examples/ssd/ssd_detect.bin \
./build/examples/ssd/SSD_300x300/deploy.prototxt \
./build/examples/ssd/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel \
./build/examples/ssd/test.txt \
--file_type image \
--out_file ./build/examples/ssd/output.txt
```
5. Verify the results  
Open `output.txt` file, you will see something like:  
```
/home/lkj/deeplearning/tmp_caffe/caffe/build/examples/ssd/dog.jpg 7 0.0255449 421 319 453 387
/home/lkj/deeplearning/tmp_caffe/caffe/build/examples/ssd/dog.jpg 9 0.0354878 388 154 452 264
... ...
```

## An analysis for SSD
