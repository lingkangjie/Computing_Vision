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

