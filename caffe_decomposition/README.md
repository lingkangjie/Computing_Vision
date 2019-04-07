# Caffe Installation
1. For ubuntu 16.04, Basic steps:
```
$ sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
$ sudo apt-get install --no-install-recommends libboost-all-dev
``` 
2. install Python, CUDA, cuDNN, BLAS/MKL
More detials, please [reference](http://caffe.berkeleyvision.org/install_apt.html)

3. make caffe
```
$ cd $CAFFE_ROOT
$ mkdir build
$ cd build
$ cmake ..
$ make all -j8
```

3. verify installation

```
$ cd $CAFFE_ROOT
$ ./data/mnist/get_mnist.sh
$ ./examples/mnist/create_mnist.sh
$ ./examples/mnist/train_lenet.sh
```
# Caffe building system profile based on CMake
1. File structure
```
│── CMakeList.txt
│── src
│    │── gtest
│    │   └── CMakeList.txt
│    └── caffe
│        └── CMakeList.txt
│── tools
│    │── CMakeList.txt
│    │── examples
│    │    └── CMakeList.txt
│    │── python
│    │    └── CMakeList.txt
│    │── matlab
│    │    └── CMakeList.txt
│    │── docs
│    │    └── CMakeList.txt
│    └── cmake
│        ├── ConfigGen.cmake
│        ├── Cuda.cmake
│        ├── Dependencies.cmake
│        ├── External
│        │   ├── gflags.cmake
│        │   └── glog.cmake
│        ├── lint.cmake
│        ├── Misc.cmake
│        ├── Modules
│        │   ├── FindAtlas.cmake
│        │   ├── FindGFlags.cmake
│        │   ├── FindGlog.cmake
│        │   ├── FindLAPACK.cmake
│        │   ├── FindLevelDB.cmake
│        │   ├── FindLMDB.cmake
│        │   ├── FindMatlabMex.cmake
│        │   ├── FindMKL.cmake
│        │   ├── FindNCCL.cmake
│        │   ├── FindNumPy.cmake
│        │   ├── FindOpenBLAS.cmake
│        │   ├── FindSnappy.cmake
│        │   └── FindvecLib.cmake
│        ├── ProtoBuf.cmake
│        ├── Summary.cmake
│        ├── Targets.cmake
│        ├── Templates
│        │   ├── CaffeConfig.cmake.in
│        │   ├── caffe_config.h.in
│        │   └── CaffeConfigVersion.cmake.in
│        ├── Uninstall.cmake.in
│        └── Utils.cmake
```
2. Notes in CMakeList.txt file
- `caffe_option` defined in `Utils.cmake`
- The `PUBLIC`, and `PRIVATE` linking in 'Dependencies.cmake'
   - When A links in B as *PRIVATE*, it is saying that A uses B in its
   implementation, but B is not used in any part of A's public API. Any code
   that makes calls into A would not need to refer directly to anything from
   B. An example of this could be a networking library A which can be built to
   use one of a number of different SSL libraries internally (which B
   represents). A presents a unified interface for client code which does not
   reference any of the internal SSL data structures or functions. Client code
   would have no idea what SSL implementation (B) is being used by A, nor does
   that client code need to care.
   - When A links in B as *INTERFACE*, it is saying that A does not use B
   in its implementation, but B is used in A's public API. Code that calls
   into A may need to refer to things from B in order to make such calls. One
   example of this is an interface library which simply forwards calls along
   to another library but doesn't actually reference the objects on the way
   through other than by a pointer or reference. Another example is where A is
   defined in CMake as an interface library, meaning it has no actual
   implementation itself, it is effectively just a collection of other
   libraries (I'm probably over-simplifying here, but you get the picture).
   - When A links in B as *PUBLIC*, it is essentially a combination of
   PRIVATE and INTERFACE. It says that A uses B in its implementation and B is
   also used in A's public API.
- The dependencies in Caffe:  
    * Boost-1.5.4: A peer-reviewed portable C++ source libraries, Caffe needs 
      `system`, `thread` and `filesystem` libraries.
    *  Threads: POSIX threads library (Wait to correct)
    * OpenMP: A Open Multi-Processing library, supports multi-platform shared 
      memory multiprocessing programming in C, C++ and Fortran.
    * Google glog and gflags: glog depends on gflags. Google Logging Library(glog)
       is a library that implements application-level logging. This library 
       provides logging APIs based on C++-style streams and various helper macros.
       gflags is a Google Commandline Flags library, and commandline flags are flags 
       that users specify on the command line when they run an executable.
    * Google protobuf: Google protocol buffers, are Google's language-neutral, 
       platform-neutral, extensible mechanism for serializing structured 
       data – think XML, but smaller, faster, and simpler. You define how you 
       want your data to be structured once, then you can use special generated 
       source code to easily write and read your structured data to and from 
       a variety of data streams and using a variety of languages.
    * HDF5: Hierarchical Data Format (HDF) is a set of *file formats* 
       (HDF4, HDF5) designed to store and organize large amounts of data. 
    * LMDB: Lightning Memory-Mapped Database (LMDB) is a software library 
       that provides a high-performance embedded transactional database in 
       the form of a key-value store. LMDB is not a relational database.
    HDF5 or LMDB? 
        (1) Reasons to use HDF5: Simple format to read/write.  
        (2) Reasons to use LMDB: LMDB uses memory-mapped files, giving much 
        better I/O performance. Works well with really large datasets. The HDF5 
        files are always read entirely into memory, so you can’t have any HDF5 file 
        exceed your memory capacity. You can easily split your data into several 
        HDF5 files though (just put several paths to h5 files in your text file). 
        Then again, compared to LMDB’s page caching the I/O performance won’t be nearly as good.
    * LevelDB: LevelDB is a fast key-value storage library written at 
        Google that provides an ordered mapping from string keys to string values.
    * Snappy: Snappy, a fast compressor/decompressor, developed by Google, open source.
        Snappy does not aim for maximum compression, or compatibility with any 
        other compression library; instead, it aims for very high speeds and reasonable compression
    * CUDA: a parallel computing platform and programming model invented by NVIDIA to use GPU.
    * NCCL: NVIDIA Collective Communications Library (NCCL) implements 
        multi-GPU and multi-node collective communication primitives that are 
        performance optimized for NVIDIA GPUs. 
    * OpenCV: A famous and well-known library for computing vision.
    * BLAS: Basic Linear Algebra Subprograms (BLAS) is a specification that 
        prescribes a set of low-level routines for performing common linear algebra 
        operations such as vector addition, scalar multiplication, dot products, 
        linear combinations, and matrix multiplication. `Atlas`, Automatically 
        Tuned Linear Algebra Software (ATLAS) is a software library for linear algebra.
        `OpenBLAS`is an optimized BLAS library. `MKL`, called Intel Math Kernel 
        Library (Intel MKL), is a library of optimized math routines for science,
        engineering, and financial applications. MKL optimizes code with minimal 
        effort for future generations of Intel processors.
    * Python, Matlab: an interface for Caffe
    * Doxygen: a tool for writing software reference documentation.
3. The dependencies in Caffe
To check the dependencies in caffe, use `graphviz` library to visualization:  
```
$ cd build
$ cmake --graphviz=caffe_dependencies.dot ..
$ dot -T png caffe_dependencies.dot -o caffe_dependencies.png
```
![caffe dependencies](./imgs/caffe_dependencies.png)
 4. CMake process for debugging
```
lkj@lkj:~/deeplearning/caffe/build$ cmake -DCMAKE_BUILD_TYPE=Debug ..
-- Boost version: 1.65.1
-- Found the following Boost libraries:
--   system
--   thread
--   filesystem
--   chrono
--   date_time
--   atomic
-- Found gflags  (include: /usr/include, library: /usr/lib/x86_64-linux-gnu/libgflags.so)
-- Found glog    (include: /usr/include, library: /usr/lib/x86_64-linux-gnu/libglog.so)
-- Found PROTOBUF Compiler: /usr/bin/protoc
-- HDF5: Using hdf5 compiler wrapper to determine C configuration
-- HDF5: Using hdf5 compiler wrapper to determine CXX configuration
-- Found lmdb    (include: /usr/include, library: /usr/lib/x86_64-linux-gnu/liblmdb.so)
-- Found LevelDB (include: /usr/include, library: /usr/lib/x86_64-linux-gnu/libleveldb.so)
-- Found Snappy  (include: /usr/include, library: /usr/lib/x86_64-linux-gnu/libsnappy.so)
CMake Warning at cmake/Dependencies.cmake:90 (message):
  -- CUDA is not detected by cmake.  Building without it...
Call Stack (most recent call first):
  CMakeLists.txt:49 (include)


-- OpenCV found (/usr/local/share/OpenCV)
-- Found Atlas (include: /usr/include/x86_64-linux-gnu library: /usr/lib/x86_64-linux-gnu/libatlas.so lapack: /usr/lib/x86_64-linux-gnu/liblapack.so
-- NumPy ver. 1.15.4 found (include: /home/lkj/.local/lib/python2.7/site-packages/numpy/core/include)
-- Boost version: 1.65.1
-- Found the following Boost libraries:
--   python
-- Detected Doxygen OUTPUT_DIRECTORY: ./doxygen/
--
-- ******************* Caffe Configuration Summary *******************
-- General:
--   Version           :   1.0.0
--   Git               :   1.0-132-g99bd9979-dirty
--   System            :   Linux
--   C++ compiler      :   /usr/bin/c++
--   Release CXX flags :   -O3 -DNDEBUG -fPIC -Wall -Wno-sign-compare -Wno-uninitialized
--   Debug CXX flags   :   -g -fPIC -Wall -Wno-sign-compare -Wno-uninitialized
--   Build type        :   Debug
--
--   BUILD_SHARED_LIBS :   ON
--   BUILD_python      :   ON
--   BUILD_matlab      :   OFF
--   BUILD_docs        :   ON
--   CPU_ONLY          :   OFF
--   USE_OPENCV        :   ON
--   USE_LEVELDB       :   ON
--   USE_LMDB          :   ON
--   USE_NCCL          :   OFF
--   ALLOW_LMDB_NOLOCK :   OFF
--   USE_HDF5          :   ON
--
-- Dependencies:
--   BLAS              :   Yes (Atlas)
--   Boost             :   Yes (ver. 1.65)
--   glog              :   Yes
--   gflags            :   Yes
--   protobuf          :   Yes (ver. 3.0.0)
--   lmdb              :   Yes (ver. 0.9.21)
--   LevelDB           :   Yes (ver. 1.20)
--   Snappy            :   Yes (ver. ..)
--   OpenCV            :   Yes (ver. 3.4.5)
--   CUDA              :   No
--
-- Python:
--   Interpreter       :   /usr/bin/python2.7 (ver. 2.7.15)
--   Libraries         :   /usr/lib/x86_64-linux-gnu/libpython2.7.so (ver 2.7.15rc1)
--   NumPy             :   /home/lkj/.local/lib/python2.7/site-packages/numpy/core/include (ver 1.15.4)
--
-- Documentaion:
--   Doxygen           :   /usr/bin/doxygen (1.8.13)
--   config_file       :   /home/lkj/deeplearning/caffe/.Doxyfile
--
-- Install:
--   Install path      :   /home/lkj/deeplearning/caffe/build/install
--
-- Configuring done
-- Generating done
-- Build files have been written to: /home/lkj/deeplearning/caffe/build
```

```
[  0%] Building CXX object src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o
[  1%] Linking CXX static library ../../lib/libcaffeproto-d.a
[  3%] Built target caffeproto
Scanning dependencies of target caffe
[  3%] Building CXX object src/caffe/CMakeFiles/caffe.dir/common.cpp.o
[  3%] Building CXX object src/caffe/CMakeFiles/caffe.dir/blob.cpp.o
[  3%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layer_factory.cpp.o
[  6%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layer.cpp.o
[  6%] Building CXX object src/caffe/CMakeFiles/caffe.dir/data_transformer.cpp.o
[  7%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/absval_layer.cpp.o
[  7%] Building CXX object src/caffe/CMakeFiles/caffe.dir/internal_thread.cpp.o
[  7%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/accuracy_layer.cpp.o
[  9%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/argmax_layer.cpp.o
[ 10%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/base_data_layer.cpp.o
[ 10%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/base_conv_layer.cpp.o
[ 10%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/batch_norm_layer.cpp.o
[ 12%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/batch_reindex_layer.cpp.o
[ 12%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/bias_layer.cpp.o
[ 14%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/bnll_layer.cpp.o
[ 15%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/contrastive_loss_layer.cpp.o
[ 15%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/clip_layer.cpp.o
[ 15%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/concat_layer.cpp.o
[ 17%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/conv_layer.cpp.o
[ 17%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/crop_layer.cpp.o
[ 18%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/cudnn_conv_layer.cpp.o
[ 18%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/cudnn_deconv_layer.cpp.o
[ 20%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/cudnn_lcn_layer.cpp.o
[ 20%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/cudnn_lrn_layer.cpp.o
[ 21%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/cudnn_pooling_layer.cpp.o
[ 21%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/cudnn_relu_layer.cpp.o
[ 23%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/cudnn_sigmoid_layer.cpp.o
[ 23%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/cudnn_softmax_layer.cpp.o
[ 25%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/cudnn_tanh_layer.cpp.o
[ 25%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/data_layer.cpp.o
[ 26%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/deconv_layer.cpp.o
[ 26%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/dropout_layer.cpp.o
[ 28%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/dummy_data_layer.cpp.o
[ 28%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/eltwise_layer.cpp.o
[ 29%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/elu_layer.cpp.o
[ 29%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/embed_layer.cpp.o
[ 31%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/euclidean_loss_layer.cpp.o
[ 31%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/exp_layer.cpp.o
[ 32%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/filter_layer.cpp.o
[ 32%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/flatten_layer.cpp.o
[ 34%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/hdf5_data_layer.cpp.o
[ 34%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/hdf5_output_layer.cpp.o
[ 34%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/hinge_loss_layer.cpp.o
[ 35%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/im2col_layer.cpp.o
[ 35%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/image_data_layer.cpp.o
[ 37%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/infogain_loss_layer.cpp.o
[ 37%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/inner_product_layer.cpp.o
[ 39%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/input_layer.cpp.o
[ 39%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/log_layer.cpp.o
[ 40%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/loss_layer.cpp.o
[ 40%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/lrn_layer.cpp.o
[ 42%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/lstm_layer.cpp.o
[ 42%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/lstm_unit_layer.cpp.o
[ 43%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/memory_data_layer.cpp.o
[ 43%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/multinomial_logistic_loss_layer.cpp.o
[ 45%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/mvn_layer.cpp.o
[ 45%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/neuron_layer.cpp.o
[ 46%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/parameter_layer.cpp.o
[ 46%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/pooling_layer.cpp.o
[ 48%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/power_layer.cpp.o
[ 48%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/prelu_layer.cpp.o
[ 50%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/recurrent_layer.cpp.o
[ 50%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/reduction_layer.cpp.o
[ 51%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/relu_layer.cpp.o
[ 51%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/reshape_layer.cpp.o
[ 53%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/rnn_layer.cpp.o
[ 53%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/scale_layer.cpp.o
[ 54%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/sigmoid_cross_entropy_loss_layer.cpp.o
[ 54%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/sigmoid_layer.cpp.o
[ 56%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/silence_layer.cpp.o
[ 56%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/slice_layer.cpp.o
[ 57%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/softmax_layer.cpp.o
[ 57%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/softmax_loss_layer.cpp.o
[ 59%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/split_layer.cpp.o
[ 59%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/spp_layer.cpp.o
[ 60%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/swish_layer.cpp.o
[ 60%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/tanh_layer.cpp.o
[ 62%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/threshold_layer.cpp.o
[ 62%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/tile_layer.cpp.o
[ 64%] Building CXX object src/caffe/CMakeFiles/caffe.dir/layers/window_data_layer.cpp.o
[ 64%] Building CXX object src/caffe/CMakeFiles/caffe.dir/net.cpp.o
[ 65%] Building CXX object src/caffe/CMakeFiles/caffe.dir/parallel.cpp.o
[ 65%] Building CXX object src/caffe/CMakeFiles/caffe.dir/solver.cpp.o
[ 65%] Building CXX object src/caffe/CMakeFiles/caffe.dir/solvers/adadelta_solver.cpp.o
[ 67%] Building CXX object src/caffe/CMakeFiles/caffe.dir/solvers/adagrad_solver.cpp.o
[ 67%] Building CXX object src/caffe/CMakeFiles/caffe.dir/solvers/adam_solver.cpp.o
[ 68%] Building CXX object src/caffe/CMakeFiles/caffe.dir/solvers/nesterov_solver.cpp.o
[ 68%] Building CXX object src/caffe/CMakeFiles/caffe.dir/solvers/rmsprop_solver.cpp.o
[ 70%] Building CXX object src/caffe/CMakeFiles/caffe.dir/solvers/sgd_solver.cpp.o
[ 73%] Building CXX object src/caffe/CMakeFiles/caffe.dir/util/cudnn.cpp.o
[ 75%] Building CXX object src/caffe/CMakeFiles/caffe.dir/util/blocking_queue.cpp.o
[ 75%] Building CXX object src/caffe/CMakeFiles/caffe.dir/syncedmem.cpp.o
[ 73%] Building CXX object src/caffe/CMakeFiles/caffe.dir/util/db_leveldb.cpp.o
[ 75%] Building CXX object src/caffe/CMakeFiles/caffe.dir/util/benchmark.cpp.o
[ 75%] Building CXX object src/caffe/CMakeFiles/caffe.dir/util/db.cpp.o
[ 73%] Building CXX object src/caffe/CMakeFiles/caffe.dir/util/db_lmdb.cpp.o
[ 76%] Building CXX object src/caffe/CMakeFiles/caffe.dir/util/hdf5.cpp.o
[ 76%] Building CXX object src/caffe/CMakeFiles/caffe.dir/util/im2col.cpp.o
[ 78%] Building CXX object src/caffe/CMakeFiles/caffe.dir/util/insert_splits.cpp.o
[ 78%] Building CXX object src/caffe/CMakeFiles/caffe.dir/util/io.cpp.o
[ 79%] Building CXX object src/caffe/CMakeFiles/caffe.dir/util/math_functions.cpp.o
[ 79%] Building CXX object src/caffe/CMakeFiles/caffe.dir/util/signal_handler.cpp.o
[ 81%] Building CXX object src/caffe/CMakeFiles/caffe.dir/util/upgrade_proto.cpp.o
[ 81%] Linking CXX shared library ../../lib/libcaffe-d.so
[ 81%] Built target caffe
[ 82%] Building CXX object tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o
[ 82%] Building CXX object tools/CMakeFiles/upgrade_net_proto_text.dir/upgrade_net_proto_text.cpp.o
[ 82%] Building CXX object tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o
[ 82%] Building CXX object tools/CMakeFiles/compute_image_mean.dir/compute_image_mean.cpp.o
[ 82%] Building CXX object tools/CMakeFiles/upgrade_net_proto_binary.dir/upgrade_net_proto_binary.cpp.o
[ 82%] Building CXX object tools/CMakeFiles/upgrade_solver_proto_text.dir/upgrade_solver_proto_text.cpp.o
[ 84%] Building CXX object tools/CMakeFiles/extract_features.dir/extract_features.cpp.o
[ 84%] Building CXX object examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o
[ 85%] Linking CXX executable compute_image_mean-d
[ 87%] Linking CXX executable convert_imageset-d
[ 87%] Built target compute_image_mean
[ 87%] Building CXX object examples/CMakeFiles/convert_mnist_data.dir/mnist/convert_mnist_data.cpp.o
[ 87%] Built target convert_imageset
[ 87%] Building CXX object examples/CMakeFiles/convert_cifar_data.dir/cifar10/convert_cifar_data.cpp.o
[ 89%] Linking CXX executable upgrade_net_proto_binary-d
[ 90%] Linking CXX executable upgrade_net_proto_text-d
[ 92%] Linking CXX executable upgrade_solver_proto_text-d
[ 92%] Linking CXX executable extract_features-d
[ 92%] Built target upgrade_net_proto_text
[ 92%] Built target upgrade_net_proto_binary
[ 92%] Built target upgrade_solver_proto_text
[ 92%] Built target extract_features
[ 92%] Building CXX object examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o
[ 93%] Building CXX object python/CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.o
[ 95%] Linking CXX executable mnist/convert_mnist_data-d
[ 96%] Linking CXX executable cifar10/convert_cifar_data-d
[ 98%] Linking CXX executable cpp_classification/classification-d
[ 98%] Built target convert_cifar_data
[ 98%] Built target convert_mnist_data
[ 98%] Linking CXX executable caffe-d
[ 98%] Built target classification
[ 98%] Built target caffe.bin
[100%] Linking CXX executable siamese/convert_mnist_siamese_data-d
[100%] Built target convert_mnist_siamese_data
[100%] Linking CXX shared library ../lib/_caffe-d.so
Creating symlink /home/lkj/deeplearning/caffe/python/caffe/_caffe.so -> /home/lkj/deeplearning/caffe/build/lib/_caffe-d.so
[100%] Built target pycaffe
```


# Caffe building system profile based on Make
1. file structrue:
```
./caffe/
    ├── caffe.cloc               
    ├── cmake
    ├── CMakeLists.txt
    ├── CONTRIBUTING.md
    ├── CONTRIBUTORS.md
    ├── data
    ├── distribute
    ├── docker
    ├── docs
    ├── examples
    ├── include
    ├── INSTALL.md
    ├── LICENSE
    ├── Makefile
    ├── Makefile.config
    ├── Makefile.config.example
    ├── matlab
    ├── models
    ├── python
    ├── README.md
    ├── scripts
    ├── src
    └── tools
```
src folder:
```
./src/
├── caffe
│   ├── blob.cpp
│   ├── CMakeLists.txt
│   ├── common.cpp
│   ├── data_transformer.cpp
│   ├── internal_thread.cpp
│   ├── layer.cpp
│   ├── layer_factory.cpp
│   │── layers
│   │    │ ... ...
│   │    │── absval_layer.cpp
│   │    │── absval_layer.cu
│   │    │── accuracy_layer.cpp
│   │    │── accuracy_layer.cu
│   │    │ ... ...
│   ├── net.cpp
│   ├── parallel.cpp
│   ├── proto
│   │     └── caffe.proto
│   ├── solver.cpp
│   ├── solvers
│   │    │ ... ...
│   │    ├── adadelta_solver.cpp
│   │    ├── adadelta_solver.cu
│   │    │ ... ...
│   ├── syncedmem.cpp
│   ├── test
│   │    ├── CMakeLists.txt
│   │    ├── test_convolution_layer.cpp
│   │    ├── test_crop_layer.cpp
│   │    | ... ...
│   │    ├── test_data
│   │    │   ├── generate_sample_data.py
│   │    │   ├── sample_data_2_gzip.h5
│   │    │   ├── sample_data.h5
│   │    │   ├── sample_data_list.txt
│   │    │   ├── solver_data.h5
│   │    │   └── solver_data_list.txt
│   │    │ ... ...
│   │    └── 
│   └── util
│       ├── benchmark.cpp
│       ├── blocking_queue.cpp
│       ├── cudnn.cpp
│       ├── db.cpp
│       ├── db_leveldb.cpp
│       ├── db_lmdb.cpp
│       ├── hdf5.cpp
│       ├── im2col.cpp
│       ├── im2col.cu
│       ├── insert_splits.cpp
│       ├── io.cpp
│       ├── math_functions.cpp
│       ├── math_functions.cu
│       ├── signal_handler.cpp
│       └── upgrade_proto.cpp
│
└── gtest
    ├── CMakeLists.txt
    ├── gtest-all.cpp
    ├── gtest.h
    └── gtest_main.cc
```
include folder:
```
./include/
└── caffe
    ├── blob.hpp
    ├── caffe.hpp
    ├── common.hpp
    ├── data_transformer.hpp
    ├── filler.hpp
    ├── internal_thread.hpp
    ├── layer_factory.hpp
    ├── layer.hpp
    ├── layers
    ├── net.hpp
    ├── parallel.hpp
    ├── sgd_solvers.hpp
    ├── solver_factory.hpp
    ├── solver.hpp
    ├── syncedmem.hpp
    ├── test
    └── util
```
1. Two files: `Makefile`, and `Makefile.config`.
2. Notes on Makefile
- As `DEBUG` maybe be set in file `Makefile.config`, so use `?=` to set
`DEBUG` when it is not already set.

# Caffe documents generated by Doxygen
[documents](https://caffe.berkeleyvision.org/doxygen/index.html)  
To generate the documentation, run `$CAFFE_ROOT/scripts/build_docs.sh`.


## Caffe files profile
Optimal: add system and global ctag files:
```
$ ctags -I __THROW –file-scope=yes –langmap=c:+.h –languages=c,c++ –links=yes –c-kinds=+p --fields=+S -R -f ~/.vim/systags /usr/include /usr/local/include

```
If generates ctags file in Caffe project root path, then `/.vimrc` setting as:
```
set hlsearch
set expandtab
set tabstop=4
set shiftwidth=4
set tags+=/home/lkj/deeplearning/caffe/tags
```

## A profile for Lenet debugging
###  Start the debug environment
```
$ gdb ./build/tools/caffe-d
$ (gdb) break main
$ (gdb) run train --solver=examples/mnist/lenet_solver.prototxt 
```


## My thoughts on C++ project
1. For a project, first we define functions ( also call features) what we want. The features
will affect how and how many the classes and functions we intend to design.
2. Design the Class, decide how many classed we want, decide the class member and behaviour,
decide how many functions we want, decide the API for function, decide which function will
call which function, etc. (Core)
3. distribute classes and functions in different files, decide which file will contain which
classes and functions
4. decide source files distribution
5. write CMake files to build
Maybe you also need to write Test code to test your project, and CMake files for Installation.


## The source file I read
1. Blob.hpp and Blob.cpp
2. math_function.hpp and math_function.cpp
3. syncedmem.hpp and syncedmem.cpp
4. layer.hpp and layer.cpp

