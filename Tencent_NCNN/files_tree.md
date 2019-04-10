# Project Files View
```
lkj@lkj:~/deeplearning/ncnn/tmp$ ls
ncnn
lkj@lkj:~/deeplearning/ncnn/tmp$ tree
.
└── ncnn
    ├── benchmark
    │   ├── alexnet.param
    │   ├── benchncnn.cpp
    │   ├── CMakeLists.txt
    │   ├── googlenet_int8.param
    │   ├── googlenet.param
    │   ├── mnasnet.param
    │   ├── mobilenet_int8.param
    │   ├── mobilenet.param
    │   ├── mobilenet_ssd_int8.param
    │   ├── mobilenet_ssd.param
    │   ├── mobilenet_v2.param
    │   ├── mobilenet_yolo.param
    │   ├── mobilenet_yolov3.param
    │   ├── proxylessnasnet.param
    │   ├── README.md
    │   ├── resnet18_int8.param
    │   ├── resnet18.param
    │   ├── resnet50_int8.param
    │   ├── resnet50.param
    │   ├── shufflenet.param
    │   ├── squeezenet_int8.param
    │   ├── squeezenet.param
    │   ├── squeezenet_ssd_int8.param
    │   ├── squeezenet_ssd.param
    │   ├── vgg16_int8.param
    │   └── vgg16.param
    ├── build.sh
    ├── CMakeLists.txt
    ├── CONTRIBUTING.md
    ├── examples
    │   ├── CMakeLists.txt
    │   ├── fasterrcnn.cpp
    │   ├── mobilenetssd.cpp
    │   ├── mobilenetv2ssdlite.cpp
    │   ├── rfcn.cpp
    │   ├── shufflenetv2.cpp
    │   ├── squeezencnn
    │   │   ├── AndroidManifest.xml
    │   │   ├── ant.properties
    │   │   ├── assets
    │   │   │   ├── squeezenet_v1.1.bin -> ../../squeezenet_v1.1.bin
    │   │   │   ├── squeezenet_v1.1.param.bin
    │   │   │   └── synset_words.txt -> ../../synset_words.txt
    │   │   ├── build.xml
    │   │   ├── jni
    │   │   │   ├── Android.mk
    │   │   │   ├── Application.mk
    │   │   │   ├── squeezencnn_jni.cpp
    │   │   │   └── squeezenet_v1.1.id.h
    │   │   ├── local.properties
    │   │   ├── proguard-project.txt
    │   │   ├── project.properties
    │   │   ├── res
    │   │   │   ├── layout
    │   │   │   │   └── main.xml
    │   │   │   └── values
    │   │   │       └── strings.xml
    │   │   └── src
    │   │       └── com
    │   │           └── tencent
    │   │               └── squeezencnn
    │   │                   ├── MainActivity.java
    │   │                   └── SqueezeNcnn.java
    │   ├── squeezenet.cpp
    │   ├── squeezenetssd.cpp
    │   ├── squeezenet_v1.1.bin
    │   ├── squeezenet_v1.1.caffemodel
    │   ├── squeezenet_v1.1.param
    │   ├── squeezenet_v1.1.prototxt
    │   ├── synset_words.txt
    │   ├── yolov2.cpp
    │   └── yolov3.cpp
    ├── images
    │   ├── 128-ncnn.png
    │   ├── 16-ncnn.png
    │   ├── 256-ncnn.png
    │   ├── 32-ncnn.png
    │   └── 64-ncnn.png
    ├── Info.plist
    ├── LICENSE.txt
    ├── package.sh
    ├── README.md
    ├── src
    │   ├── allocator.cpp
    │   ├── allocator.h
    │   ├── benchmark.cpp
    │   ├── benchmark.h
    │   ├── blob.cpp
    │   ├── blob.h
    │   ├── CMakeLists.txt
    │   ├── command.cpp
    │   ├── command.h
    │   ├── cpu.cpp
    │   ├── cpu.h
    │   ├── gpu.cpp
    │   ├── gpu.h
    │   ├── layer
    │   │   ├── absval.cpp
    │   │   ├── absval.h
    │   │   ├── argmax.cpp
    │   │   ├── argmax.h
    │   │   ├── arm
    │   │   │   ├── absval_arm.cpp
    │   │   │   ├── absval_arm.h
    │   │   │   ├── batchnorm_arm.cpp
    │   │   │   ├── batchnorm_arm.h
    │   │   │   ├── bias_arm.cpp
    │   │   │   ├── bias_arm.h
    │   │   │   ├── clip_arm.cpp
    │   │   │   ├── clip_arm.h
    │   │   │   ├── convolution_1x1.h
    │   │   │   ├── convolution_1x1_int8.h
    │   │   │   ├── convolution_2x2.h
    │   │   │   ├── convolution_3x3.h
    │   │   │   ├── convolution_3x3_int8.h
    │   │   │   ├── convolution_4x4.h
    │   │   │   ├── convolution_5x5.h
    │   │   │   ├── convolution_5x5_int8.h
    │   │   │   ├── convolution_7x7.h
    │   │   │   ├── convolution_7x7_int8.h
    │   │   │   ├── convolution_arm.cpp
    │   │   │   ├── convolution_arm.h
    │   │   │   ├── convolutiondepthwise_3x3.h
    │   │   │   ├── convolutiondepthwise_3x3_int8.h
    │   │   │   ├── convolutiondepthwise_5x5.h
    │   │   │   ├── convolutiondepthwise_arm.cpp
    │   │   │   ├── convolutiondepthwise_arm.h
    │   │   │   ├── convolution_sgemm_int8.h
    │   │   │   ├── deconvolution_3x3.h
    │   │   │   ├── deconvolution_4x4.h
    │   │   │   ├── deconvolution_arm.cpp
    │   │   │   ├── deconvolution_arm.h
    │   │   │   ├── deconvolutiondepthwise_arm.cpp
    │   │   │   ├── deconvolutiondepthwise_arm.h
    │   │   │   ├── dequantize_arm.cpp
    │   │   │   ├── dequantize_arm.h
    │   │   │   ├── eltwise_arm.cpp
    │   │   │   ├── eltwise_arm.h
    │   │   │   ├── innerproduct_arm.cpp
    │   │   │   ├── innerproduct_arm.h
    │   │   │   ├── interp_arm.cpp
    │   │   │   ├── interp_arm.h
    │   │   │   ├── lrn_arm.cpp
    │   │   │   ├── lrn_arm.h
    │   │   │   ├── neon_mathfun.h
    │   │   │   ├── pooling_2x2.h
    │   │   │   ├── pooling_3x3.h
    │   │   │   ├── pooling_arm.cpp
    │   │   │   ├── pooling_arm.h
    │   │   │   ├── prelu_arm.cpp
    │   │   │   ├── prelu_arm.h
    │   │   │   ├── quantize_arm.cpp
    │   │   │   ├── quantize_arm.h
    │   │   │   ├── relu_arm.cpp
    │   │   │   ├── relu_arm.h
    │   │   │   ├── requantize_arm.cpp
    │   │   │   ├── requantize_arm.h
    │   │   │   ├── scale_arm.cpp
    │   │   │   ├── scale_arm.h
    │   │   │   ├── sigmoid_arm.cpp
    │   │   │   ├── sigmoid_arm.h
    │   │   │   ├── softmax_arm.cpp
    │   │   │   └── softmax_arm.h
    │   │   ├── batchnorm.cpp
    │   │   ├── batchnorm.h
    │   │   ├── bias.cpp
    │   │   ├── bias.h
    │   │   ├── binaryop.cpp
    │   │   ├── binaryop.h
    │   │   ├── bnll.cpp
    │   │   ├── bnll.h
    │   │   ├── cast.cpp
    │   │   ├── cast.h
    │   │   ├── clip.cpp
    │   │   ├── clip.h
    │   │   ├── concat.cpp
    │   │   ├── concat.h
    │   │   ├── convolution.cpp
    │   │   ├── convolutiondepthwise.cpp
    │   │   ├── convolutiondepthwise.h
    │   │   ├── convolution.h
    │   │   ├── crop.cpp
    │   │   ├── crop.h
    │   │   ├── deconvolution.cpp
    │   │   ├── deconvolutiondepthwise.cpp
    │   │   ├── deconvolutiondepthwise.h
    │   │   ├── deconvolution.h
    │   │   ├── dequantize.cpp
    │   │   ├── dequantize.h
    │   │   ├── detectionoutput.cpp
    │   │   ├── detectionoutput.h
    │   │   ├── dropout.cpp
    │   │   ├── dropout.h
    │   │   ├── eltwise.cpp
    │   │   ├── eltwise.h
    │   │   ├── elu.cpp
    │   │   ├── elu.h
    │   │   ├── embed.cpp
    │   │   ├── embed.h
    │   │   ├── expanddims.cpp
    │   │   ├── expanddims.h
    │   │   ├── exp.cpp
    │   │   ├── exp.h
    │   │   ├── flatten.cpp
    │   │   ├── flatten.h
    │   │   ├── innerproduct.cpp
    │   │   ├── innerproduct.h
    │   │   ├── input.cpp
    │   │   ├── input.h
    │   │   ├── instancenorm.cpp
    │   │   ├── instancenorm.h
    │   │   ├── interp.cpp
    │   │   ├── interp.h
    │   │   ├── log.cpp
    │   │   ├── log.h
    │   │   ├── lrn.cpp
    │   │   ├── lrn.h
    │   │   ├── lstm.cpp
    │   │   ├── lstm.h
    │   │   ├── memorydata.cpp
    │   │   ├── memorydata.h
    │   │   ├── mvn.cpp
    │   │   ├── mvn.h
    │   │   ├── normalize.cpp
    │   │   ├── normalize.h
    │   │   ├── packing.cpp
    │   │   ├── packing.h
    │   │   ├── padding.cpp
    │   │   ├── padding.h
    │   │   ├── permute.cpp
    │   │   ├── permute.h
    │   │   ├── pooling.cpp
    │   │   ├── pooling.h
    │   │   ├── power.cpp
    │   │   ├── power.h
    │   │   ├── prelu.cpp
    │   │   ├── prelu.h
    │   │   ├── priorbox.cpp
    │   │   ├── priorbox.h
    │   │   ├── proposal.cpp
    │   │   ├── proposal.h
    │   │   ├── psroipooling.cpp
    │   │   ├── psroipooling.h
    │   │   ├── quantize.cpp
    │   │   ├── quantize.h
    │   │   ├── reduction.cpp
    │   │   ├── reduction.h
    │   │   ├── relu.cpp
    │   │   ├── relu.h
    │   │   ├── reorg.cpp
    │   │   ├── reorg.h
    │   │   ├── requantize.cpp
    │   │   ├── requantize.h
    │   │   ├── reshape.cpp
    │   │   ├── reshape.h
    │   │   ├── rnn.cpp
    │   │   ├── rnn.h
    │   │   ├── roialign.cpp
    │   │   ├── roialign.h
    │   │   ├── roipooling.cpp
    │   │   ├── roipooling.h
    │   │   ├── scale.cpp
    │   │   ├── scale.h
    │   │   ├── shader
    │   │   │   ├── absval.comp
    │   │   │   ├── absval_pack4.comp
    │   │   │   ├── batchnorm.comp
    │   │   │   ├── batchnorm_pack4.comp
    │   │   │   ├── binaryop.comp
    │   │   │   ├── binaryop_pack4.comp
    │   │   │   ├── cast_fp16_to_fp32.comp
    │   │   │   ├── cast_fp16_to_fp32_pack4.comp
    │   │   │   ├── cast_fp32_to_fp16.comp
    │   │   │   ├── cast_fp32_to_fp16_pack4.comp
    │   │   │   ├── clip.comp
    │   │   │   ├── clip_pack4.comp
    │   │   │   ├── concat.comp
    │   │   │   ├── concat_pack4.comp
    │   │   │   ├── concat_pack4to1.comp
    │   │   │   ├── convolution_1x1s1d1.comp
    │   │   │   ├── convolution.comp
    │   │   │   ├── convolutiondepthwise.comp
    │   │   │   ├── convolutiondepthwise_group.comp
    │   │   │   ├── convolutiondepthwise_group_pack1to4.comp
    │   │   │   ├── convolutiondepthwise_group_pack4.comp
    │   │   │   ├── convolutiondepthwise_group_pack4to1.comp
    │   │   │   ├── convolutiondepthwise_pack4.comp
    │   │   │   ├── convolution_pack1to4.comp
    │   │   │   ├── convolution_pack4.comp
    │   │   │   ├── convolution_pack4to1.comp
    │   │   │   ├── crop.comp
    │   │   │   ├── crop_pack4.comp
    │   │   │   ├── deconvolution.comp
    │   │   │   ├── deconvolutiondepthwise.comp
    │   │   │   ├── deconvolutiondepthwise_group.comp
    │   │   │   ├── deconvolutiondepthwise_group_pack1to4.comp
    │   │   │   ├── deconvolutiondepthwise_group_pack4.comp
    │   │   │   ├── deconvolutiondepthwise_group_pack4to1.comp
    │   │   │   ├── deconvolutiondepthwise_pack4.comp
    │   │   │   ├── deconvolution_pack1to4.comp
    │   │   │   ├── deconvolution_pack4.comp
    │   │   │   ├── deconvolution_pack4to1.comp
    │   │   │   ├── dropout.comp
    │   │   │   ├── dropout_pack4.comp
    │   │   │   ├── eltwise.comp
    │   │   │   ├── eltwise_pack4.comp
    │   │   │   ├── flatten.comp
    │   │   │   ├── flatten_pack4.comp
    │   │   │   ├── innerproduct.comp
    │   │   │   ├── innerproduct_pack1to4.comp
    │   │   │   ├── innerproduct_pack4.comp
    │   │   │   ├── innerproduct_pack4to1.comp
    │   │   │   ├── interp_bicubic_coeffs.comp
    │   │   │   ├── interp_bicubic.comp
    │   │   │   ├── interp_bicubic_pack4.comp
    │   │   │   ├── interp.comp
    │   │   │   ├── interp_pack4.comp
    │   │   │   ├── lrn_norm_across_channel_pack4.comp
    │   │   │   ├── lrn_norm.comp
    │   │   │   ├── lrn_norm_within_channel_pack4.comp
    │   │   │   ├── lrn_square_pad_across_channel_pack4.comp
    │   │   │   ├── lrn_square_pad.comp
    │   │   │   ├── lrn_square_pad_within_channel_pack4.comp
    │   │   │   ├── packing_1to4.comp
    │   │   │   ├── packing_4to1.comp
    │   │   │   ├── padding.comp
    │   │   │   ├── padding_pack4.comp
    │   │   │   ├── permute.comp
    │   │   │   ├── permute_pack4to1.comp
    │   │   │   ├── pooling.comp
    │   │   │   ├── pooling_global.comp
    │   │   │   ├── pooling_global_pack4.comp
    │   │   │   ├── pooling_pack4.comp
    │   │   │   ├── prelu.comp
    │   │   │   ├── prelu_pack4.comp
    │   │   │   ├── priorbox.comp
    │   │   │   ├── priorbox_mxnet.comp
    │   │   │   ├── relu.comp
    │   │   │   ├── relu_pack4.comp
    │   │   │   ├── reorg.comp
    │   │   │   ├── reorg_pack1to4.comp
    │   │   │   ├── reorg_pack4.comp
    │   │   │   ├── reshape.comp
    │   │   │   ├── reshape_pack1to4.comp
    │   │   │   ├── reshape_pack4.comp
    │   │   │   ├── reshape_pack4to1.comp
    │   │   │   ├── scale.comp
    │   │   │   ├── scale_pack4.comp
    │   │   │   ├── shufflechannel.comp
    │   │   │   ├── shufflechannel_pack4.comp
    │   │   │   ├── sigmoid.comp
    │   │   │   ├── sigmoid_pack4.comp
    │   │   │   ├── softmax_div_sum.comp
    │   │   │   ├── softmax_div_sum_pack4.comp
    │   │   │   ├── softmax_exp_sub_max.comp
    │   │   │   ├── softmax_exp_sub_max_pack4.comp
    │   │   │   ├── softmax_reduce_max.comp
    │   │   │   ├── softmax_reduce_max_pack4.comp
    │   │   │   ├── softmax_reduce_sum.comp
    │   │   │   ├── softmax_reduce_sum_pack4.comp
    │   │   │   ├── tanh.comp
    │   │   │   ├── tanh_pack4.comp
    │   │   │   ├── unaryop.comp
    │   │   │   └── unaryop_pack4.comp
    │   │   ├── shufflechannel.cpp
    │   │   ├── shufflechannel.h
    │   │   ├── sigmoid.cpp
    │   │   ├── sigmoid.h
    │   │   ├── slice.cpp
    │   │   ├── slice.h
    │   │   ├── softmax.cpp
    │   │   ├── softmax.h
    │   │   ├── split.cpp
    │   │   ├── split.h
    │   │   ├── spp.cpp
    │   │   ├── spp.h
    │   │   ├── squeeze.cpp
    │   │   ├── squeeze.h
    │   │   ├── tanh.cpp
    │   │   ├── tanh.h
    │   │   ├── threshold.cpp
    │   │   ├── threshold.h
    │   │   ├── tile.cpp
    │   │   ├── tile.h
    │   │   ├── unaryop.cpp
    │   │   ├── unaryop.h
    │   │   ├── x86
    │   │   │   ├── avx_mathfun.h
    │   │   │   ├── convolution_1x1.h
    │   │   │   ├── convolution_1x1_int8.h
    │   │   │   ├── convolution_3x3.h
    │   │   │   ├── convolution_3x3_int8.h
    │   │   │   ├── convolution_5x5.h
    │   │   │   ├── convolution_5x5_int8.h
    │   │   │   ├── convolution_7x7_int8.h
    │   │   │   ├── convolutiondepthwise_3x3.h
    │   │   │   ├── convolutiondepthwise_3x3_int8.h
    │   │   │   ├── convolutiondepthwise_x86.cpp
    │   │   │   ├── convolutiondepthwise_x86.h
    │   │   │   ├── convolution_sgemm_int8.h
    │   │   │   ├── convolution_x86.cpp
    │   │   │   ├── convolution_x86.h
    │   │   │   └── sse_mathfun.h
    │   │   ├── yolodetectionoutput.cpp
    │   │   ├── yolodetectionoutput.h
    │   │   ├── yolov3detectionoutput.cpp
    │   │   └── yolov3detectionoutput.h
    │   ├── layer.cpp
    │   ├── layer_declaration.h.in
    │   ├── layer.h
    │   ├── layer_registry.h.in
    │   ├── layer_shader_registry.h.in
    │   ├── layer_shader_spv_data.h.in
    │   ├── layer_type_enum.h.in
    │   ├── layer_type.h
    │   ├── mat.cpp
    │   ├── mat.h
    │   ├── mat_pixel.cpp
    │   ├── mat_pixel_resize.cpp
    │   ├── modelbin.cpp
    │   ├── modelbin.h
    │   ├── net.cpp
    │   ├── net.h
    │   ├── opencv.cpp
    │   ├── opencv.h
    │   ├── paramdict.cpp
    │   ├── paramdict.h
    │   ├── pipeline.cpp
    │   ├── pipeline.h
    │   └── platform.h.in
    ├── toolchains
    │   ├── aarch64-linux-gnu.toolchain.cmake
    │   ├── arm-linux-gnueabihf.toolchain.cmake
    │   ├── arm-linux-gnueabi.toolchain.cmake
    │   ├── himix100.toolchain.cmake
    │   ├── hisiv300.toolchain.cmake
    │   ├── hisiv500.toolchain.cmake
    │   ├── host.gcc.toolchain.cmake
    │   ├── iossimxc.toolchain.cmake
    │   ├── iossimxc-x64.toolchain.cmake
    │   ├── ios.toolchain.cmake
    │   ├── iosxc-arm64.toolchain.cmake
    │   ├── iosxc.toolchain.cmake
    │   └── pi3.toolchain.cmake
    └── tools
        ├── caffe
        │   ├── caffe2ncnn.cpp
        │   ├── caffe.proto
        │   └── CMakeLists.txt
        ├── CMakeLists.txt
        ├── darknet
        │   └── readme.txt
        ├── mxnet
        │   ├── CMakeLists.txt
        │   └── mxnet2ncnn.cpp
        ├── ncnn2mem.cpp
        ├── onnx
        │   ├── CMakeLists.txt
        │   ├── onnx2ncnn.cpp
        │   └── onnx.proto
        ├── plugin
        │   ├── ImageWatchNCNN.natvis
        │   ├── README.md
        │   └── snapshot.png
        ├── pytorch
        │   └── readme.txt
        └── tensorflow
            ├── attr_value.proto
            ├── CMakeLists.txt
            ├── function.proto
            ├── graph.proto
            ├── node_def.proto
            ├── op_def.proto
            ├── resource_handle.proto
            ├── tensorflow2ncnn.cpp
            ├── tensor.proto
            ├── tensor_shape.proto
            ├── types.proto
            └── versions.proto

28 directories, 453 files
```
