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
# Caffe profile for CMakeList.txt
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
- The dependencies in Caffe
1. Boost-1.5.4: A peer-reviewed portable C++ source libraries, Caffe needs 
  `system`, `thread` and `filesystem` libraries.
2. Threads: POSIX threads library (Wait to correct)
3. OpenMP: A Open Multi-Processing library, supports multi-platform shared 
  memory multiprocessing programming in C, C++ and Fortran.
4. Google glog and gflags: glog depends on gflags. Google Logging Library(glog)
   is a library that implements application-level logging. This library 
   provides logging APIs based on C++-style streams and various helper macros.
   gflags is a Google Commandline Flags library, and commandline flags are flags 
   that users specify on the command line when they run an executable.
5. Google protobuf: Google protocol buffers, are Google's language-neutral, 
   platform-neutral, extensible mechanism for serializing structured 
   data – think XML, but smaller, faster, and simpler. You define how you 
   want your data to be structured once, then you can use special generated 
   source code to easily write and read your structured data to and from 
   a variety of data streams and using a variety of languages.
6. HDF5: Hierarchical Data Format (HDF) is a set of *file formats* 
   (HDF4, HDF5) designed to store and organize large amounts of data. 
7. LMDB: Lightning Memory-Mapped Database (LMDB) is a software library 
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
8. LevelDB: LevelDB is a fast key-value storage library written at 
Google that provides an ordered mapping from string keys to string values.
9. Snappy: Snappy, a fast compressor/decompressor, developed by Google, open source.
Snappy does not aim for maximum compression, or compatibility with any 
other compression library; instead, it aims for very high speeds and reasonable compression
10. CUDA: a parallel computing platform and programming model invented by NVIDIA to use GPU.
11. NCCL: NVIDIA Collective Communications Library (NCCL) implements 
multi-GPU and multi-node collective communication primitives that are 
performance optimized for NVIDIA GPUs. 
12. OpenCV: A famous and well-known library for computing vision.
13. BLAS: Basic Linear Algebra Subprograms (BLAS) is a specification that 
prescribes a set of low-level routines for performing common linear algebra 
operations such as vector addition, scalar multiplication, dot products, 
linear combinations, and matrix multiplication. `Atlas`, Automatically 
Tuned Linear Algebra Software (ATLAS) is a software library for linear algebra.
`OpenBLAS`is an optimized BLAS library. `MKL`, called Intel Math Kernel 
Library (Intel MKL) is a library of optimized math routines for science,
engineering, and financial applications. MKL optimizes code with minimal 
effort for future generations of Intel processors.
14. Python, Matlab: an interface for Caffe
15. Doxygen: a tool for writing software reference documentation.
