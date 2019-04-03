#!/usr/bin/env python
# draw the dependencies in Caffe library

from graphviz import Digraph

#u = Digraph('unix', filename='caffe_dependencies',format='svg')
u = Digraph('unix', filename='caffe_dependencies',format='pdf')
u.attr(size='6,6')
u.node_attr.update(color='lightblue2', style='filled')

# blob.cpp
u.edge('climits','blob.cpp')
u.edge('vector','blob.cpp')
u.edge('blob.hpp','blob.cpp')
u.edge('common.hpp','blob.cpp')
u.edge('syncedmen.hpp','blob.cpp')
u.edge('util/math_functions.hpp','blob.cpp')

# common.cpp
u.edge('boost/thread.hpp','common.cpp')
u.edge('glog/logging.h','common.cpp')
u.edge('cmath','common.cpp')
u.edge('ctime','common.cpp')
u.edge('caffe/common.hpp','common.cpp')
u.edge('caffe/util/rng.hpp','common.cpp')

# data_transformer.cpp
u.edge('opencv2/core/core.hpp','data_transformer.cpp')
u.edge('string','data_transformer.cpp')
u.edge('vector','data_transformer.cpp')
u.edge('caffe/data_transformer.hpp','data_transformer.cpp')
u.edge('caffe/util/io.hpp','data_transformer.cpp')
u.edge('caffe/util/math_functions.hpp','data_transformer.cpp')
u.edge('caffe/util/rng.hpp','data_transformer.cpp')

# internal_thread.cpp
u.edge('boost/thread.hpp','internal_thread.cpp')
u.edge('exception','internal_thread.cpp')
u.edge('caffe/internal_thread.hpp','internal_thread.cpp')
u.edge('caffe/util/math_functions.hpp','internal_thread.cpp')

# layer.cpp
u.edge('caffe/layer.hpp','layer.cpp')

# layer_factory.cpp
u.edge('boost/python.hpp','layer_factory.cpp')
u.edge('string','layer_factory.cpp')
u.edge('caffe/layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/clip_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/conv_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/deconv_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/lrn_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/pooling_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/relu_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/sigmoid_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/softmax_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/tanh_layer.hpp','layer_factory.hpp')
u.edge('caffe/proto/caffe.pb.h','layer_factory.hpp')

u.edge('caffe/layers/cudnn_conv_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/cudnn_deconv_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/cudnn_lcn_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/cudnn_lrn_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/cudnn_pooling_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/cudnn_relu_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/cudnn_sigmoid_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/cudnn_softmax_layer.hpp','layer_factory.hpp')
u.edge('caffe/layers/cudnn_tanh_layer.hpp','layer_factory.hpp')

# net.cpp
u.edge('algorithm','net.cpp')
u.edge('map','net.cpp')
u.edge('set','net.cpp')
u.edge('string','net.cpp')
u.edge('utility','net.cpp')
u.edge('vector','net.cpp')
u.edge('hdf5.h','net.cpp')
u.edge('caffe/common.hpp','net.cpp')
u.edge('caffe/layer.hpp','net.cpp')
u.edge('caffe/net.hpp','net.cpp')
u.edge('caffe/parallel.hpp','net.cpp')
u.edge('caffe/proto/caffe.pb.h','net.cpp')
u.edge('caffe/util/hdf5.hpp','net.cpp')

# parallel.cpp
u.edge('cuda_runtime.h','parallel.cpp')
u.edge('glog/logging.h','parallel.cpp')
u.edge('stdio.h','parallel.cpp')
u.edge('sstream','parallel.cpp')
u.edge('string','parallel.cpp')
u.edge('vector','parallel.cpp')
u.edge('caffe/caffe.hpp','parallel.cpp')
u.edge('caffe/parallel.hpp','parallel.cpp')
u.edge('caffe/sgd_solvers.hpp','parallel.cpp')

# solver.cpp
u.edge('cstdio','solver.cpp')
u.edge('string','solver.cpp')
u.edge('vector','solver.cpp')
u.edge('boost/algorithm/string.hpp','solver.cpp')
u.edge('caffe/solver.hpp','solver.cpp')
u.edge('caffe/util/format.hpp','solver.cpp')
u.edge('caffe/util/hdf5.hpp','solver.cpp')
u.edge('caffe/util/io.hpp','solver.cpp')
u.edge('caffe/util/upgrade_proto.hpp','solver.cpp')

# syncedmem.cpp
u.edge('caffe/common.hpp','syncedmen.cpp')
u.edge('caffe/syncedmen.hpp','syncedmen.cpp')
u.edge('caffe/util/math_functions.hpp','syncedmen.cpp')

u.view()
