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
* Draw network structure  
```
$ cd SSD_CAFFE_ROOT
$ python ./python/draw_net.py ./build/examples/ssd/SSD_300x300/train.prototxt SSD_structure.png
```
There are some common error for drawing:  
  * `'google.protobuf.pyext._message.RepeatedScalarConta' object has no attribute '_values'`.
Solution: install protobuf < 3.0.0, such as `sudo pip uninstall protobuf`, `sudo pip install protobuf==2.6.1`
  * `TypeError: __init__() got an unexpected keyword argument 'syntax'`
Solution: I use `vim ./python/caffe/proto/caffe_pb2.py` and comment all `syntax='proto2'`.

* Learning by Training  
Follow the official recommand step, prepare the data. There are some suggestions:  
  * Don't Install Pycaffe. As always build different type of Caffe version, e.g. SSD Caffe, official Caffe et al.
Every time we want Pycaffe, We just give Python a path to search the Pycaffe 
we have builded by add the follow two statements to Python sctript run:  
```
caffe_path = '/home/lkj/deeplearning/tmp_caffe/caffe/python'
sys.path.insert(0,caffe_path)
```
  * Use Python 2 may be well. 
* After data prepared done, you will see:   
```
lkj@lkj:~/deeplearning/tmp_caffe/caffe$ ls -lh /home/lkj/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb/
total 1.7G
-rw-rw-r-- 1 lkj lkj 1.7G Apr 11 16:55 data.mdb
-rw-rw-r-- 1 lkj lkj 8.0K Apr 11 16:55 lock.mdb
lkj@lkj:~/deeplearning/tmp_caffe/caffe$ ls -lh /home/lkj/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb/
total 426M
-rw-rw-r-- 1 lkj lkj 426M Apr 11 16:46 data.mdb
-rw-rw-r-- 1 lkj lkj 8.0K Apr 11 16:46 lock.mdb
```




