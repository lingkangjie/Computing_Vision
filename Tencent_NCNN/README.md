# Source Code Reading for Tencent NCNN Libarary
## Installation
```
$ git clone https://github.com/Tencent/ncnn.git
$ cd ncnn
$ mkdir -p build
$ cd build
$ cmake ..
$ make -j4
```
## An Example
1. copy or download three files: `squeezenet_v1.1.caffemodel`,
`squeezenet_v1.1.param`, and `synset_words.txt`, and put it in
`./build/examples` folder.
2. run
```
lkj@lkj:~/deeplearning/ncnn/build/examples$ ./squeezenet dog.jpg
263 = 0.863361
264 = 0.126785
151 = 0.006212
```
3. verify the results:
In `synset_words.txt`, there are total 1000 classes. `vim synset_word.txt`, you will see:
```
 ... ... 
 151 n02077923 sea lion
 ... ...
 263 n02112706 Brabancon griffon
 264 n02113023 Pembroke, Pembroke Welsh corgi
 ... ...
```
## A Deeper example
1. Download [MobileNetSSD_deploy.caffemodel](https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view), `MobileNetSSD_deploy.prototxt` two files
2. Update `*.caffemodel` and `*.prototxt` files, as NCNN only recognizes new file format.
Build SSD Caffe:  
```
$ git clone https://github.com/weiliu89/caffe.git
$ cd caffe
$ git checkout ssd
$ mkdir -p build
$ cd build
$ cmake ..
$ make all -j8
```
Convert old files to new files:  
```
lkj@lkj:~/deeplearning/ncnn$ ../tmp_caffe/caffe/build/tools/upgrade_net_proto_text ./build/examples/MobileNetSSD_deploy.prototxt ./build/exampl
es/MobileNetSSD_deploy_new.prototxt
I0410 09:24:41.075047 47159 upgrade_proto.cpp:67] Attempting to upgrade input file specified using deprecated input fields: ./build/examples/MobileNetSSD_deploy.prototxt
I0410 09:24:41.075701 47159 upgrade_proto.cpp:70] Successfully upgraded file specified using deprecated input fields.
W0410 09:24:41.075731 47159 upgrade_proto.cpp:72] Note that future Caffe releases will only support input layers and not input fields.
I0410 09:24:41.095029 47159 upgrade_net_proto_text.cpp:49] Wrote upgraded NetParameter text proto to ./build/examples/MobileNetSSD_deploy_new.prototxt
lkj@lkj:~/deeplearning/ncnn$ ../tmp_caffe/caffe/build/tools/upgrade_net_proto_binary ./build/examples/MobileNetSSD_deploy.caffemodel ./build/ex
amples/MobileNetSSD_deploy_new.caffemodel
E0410 09:26:09.719741 48152 upgrade_net_proto_binary.cpp:43] File already in latest proto format: ./build/examples/MobileNetSSD_deploy.caffemodel
I0410 09:26:09.810542 48152 upgrade_net_proto_binary.cpp:48] Wrote upgraded NetParameter binary proto to ./build/examples/MobileNetSSD_deploy_new.caffemodel
```
3. Confirm the input image number, the first `dim=1`:  
```
name: "MobileNet-SSD"
layer {
  name: "input"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 300
      dim: 300
    }
  }
}
```
4. Convert Caffe model to NCNN model:  
`*.caffemodel` ---> `*.bin`, `*.prototxt` ---> `*.param`   
```
lkj@lkj:~/deeplearning/ncnn$ ./build/tools/caffe/caffe2ncnn ./build/examples/MobileNetSSD_deploy_new.prototxt ./build/examples/MobileNetSSD_deploy_new.caffemodel ./build/examples/MobileNetSSD_deploy.param ./build/examples/MobileNetSSD_deploy.bin
```
5. (optional) Encryption 
```
lkj@lkj:~/deeplearning/ncnn$ ./build/tools/ncnn2mem
Usage: ./build/tools/ncnn2mem [ncnnproto] [ncnnbin] [idcpppath] [memcpppath]
lkj@lkj:~/deeplearning/ncnn$ ./build/tools/ncnn2mem ./build/examples/MobileNetSSD_deploy.param ./build/examples/MobileNetSSD_deploy.bin ./build/examples/MobileNetSSD_deploy.id.h ./build/examples/MobileNetSSD_deploy.men.h
lkj@lkj:~/deeplearning/ncnn$ mv MobileNetSSD_deploy.param.bin ./build/examples/
```
command `vimdiff MobileNetSSD_deploy.param.bin MobileNetSSD_deploy.param` and `:%!xxd` to check they are different:  
```
  00000000: dd85 7600 7f00 0000 9600 0000 1000 0000  ..v.............  |  00000000: 3737 3637 3531 370a 3132 3720 3135 300a  7767517.127 150.
  00000010: 0000 0000 0100 0000 0000 0000 0000 0000  ................  |  00000010: 496e 7075 7420 2020 2020 2020 2020 2020  Input
  00000020: 2c01 0000 0100 0000 2c01 0000 0200 0000  ,.......,.......  |  00000020: 2069 6e70 7574 2020 2020 2020 2020 2020   input
  00000030: 0300 0000 17ff ffff 2100 0000 0100 0000  ........!.......  |  00000030: 2020 3020 3120 6461 7461 2030 3d33 3030    0 1 data 0=300
  00000040: 0700 0000 0000 0000 0100 0000 0200 0000  ................  |  00000040: 2031 3d33 3030 2032 3d33 0a53 706c 6974   1=300 2=3.Split
  00000050: 0300 0000 0400 0000 0500 0000 0600 0000  ................  |  00000050: 2020 2020 2020 2020 2020 2020 7370 6c69              spli
```
6. Edit Sourece Code
`cp ./examples/mobilenetssd.cpp ./examples/MobileNetSSD.cpp` and `vim ./examples/MobileNetSSD.cpp, and modify to:  
```
... ...
    // if have encrypted, use "MobileNetSSD_deploy.param.bin"
    mobilenet.load_param("MobileNetSSD_deploy.param");
    mobilenet.load_model("MobileNetSSD_deploy.bin");
... ...
```
And copy two files to examples folder:  
```
lkj@lkj:~/deeplearning/ncnn/examples$ cp ../build/examples/MobileNetSSD_deploy.param ./
lkj@lkj:~/deeplearning/ncnn/examples$ cp ../build/examples/MobileNetSSD_deploy.bin ./
```
7. Edit CMake files:  
Add
```
add_executable(MobileNetSSD MobileNetSSD.cpp)
target_link_libraries(MobileNetSSD${NCNN_EXAMPLE_LINK_LIBRARIES})
```
to `./examples/CMakeLists.txt`
Uncomment `vim NCNN_ROOT/CMakeLists.txt` line 93 `add_subdirectory(examples)`
8. Build again
```
$ cd NCNN_ROOT
$ cd ./build
$ cmake ..
$ make -j4
```
9. Verify the result
```
$ cd NCNN_ROOT/build/examples/
$ ./MobileNetSSD dog.jpg
```
