## An Example for Debug
### data preparation
[reference](https://pjreddie.com/darknet/train-cifar/)
```
$ git clone https://github.com/pjreddie/darknet
$ cd darknet
$ make
$ cd data
$ wget https://pjreddie.com/media/files/cifar.tgz
$ tar xzf cifar.tgz
$ cat cifar/labels.txt
$ cd cifar
$ lkj@lkj:~/Computing_Vision/darknet/data/cifar$ find /home/lkj/Computing_Vision/darknet/data/cifar/train/ -name \*.png > train.list
$ lkj@lkj:~/Computing_Vision/darknet/data/cifar$ find /home/lkj/Computing_Vision/darknet/data/cifar/test/ -name \*.png > test.list
$ cd ../..
```
As I has prepared a small data set for debugging, you do not need to download the original Cifar dataset. The network configuration file is located in `./cgf/cifar_small.cgf`, and `./cfg/cifar.data`.  
Set `DEBUG=1` in Makefile, and make again.
```
$ gdb ./darknet
(gdb) b main
(gdb) run classifier train ./cfg/cifar.data ./cfg/cifar_small.cfg
```
### Training Processes
1. call `./examples/darknet.c`, call function `classifier.c:run_classifier()` in `darknet.c:449`.
2. in `classifier.c:run_classifier()`, after some arguments pasering, call `classifer.c:train_classifier()`.
3. in `classifer.c:train_classifier()`, first call `./src/network.c:load_network()`-> `./src/network.c:parse_network_cgf()`.
`clear` var is whether to clear pre-weights in network. in `parse_network_cgf()--call-->network* make_network(int n)`, here n=11, as there are total 11 layers in `cifar_small.cgf`. The return for `make_network()` is a struct `network` defined in `./include/darknet.h:430`.
4. after `make_network()` return, set network arguments from `cifar_small.cgf` using `parse_network_cgf():parse_net_options()`. That is to say, the original return OBJ from `make_network()` contains some empty values, e.g. batch=0, momentum=0 et al.
5. after set network param, we set `size_params` defined in `./src/parser.c`, a more higher level param wraper for training.
6. last in `parse_network_cgf()`, we constructure layer by layer through parsing, at each parsed layer will return the corresponding layer struct such as `maxpool_layer` data struct defined in `./src/maxpool_layer.h:maxpool_layer` inherited in `./include/darknet.h`, and print the layer configurarions out. After all layers are constructured done, return `net` data structure. (TODO: what is `workspace_size` meaning?)
7. When we return to `./src/network.c:load_network()`, we get a network struct `net`, then set or clear pre-trained weights, and return `net`. In `network` structure, containes many `layers`. 
8. Then read `./cfg/cifar.data` file, parse it, code snippet from `classifer.c:44` to `classifer.c:62`.
9. Set `load_args args` defined in `darknet.h`, code snippet from 64 to 89. This parms used to image propressing and fetching.
10. Pass `args` to `load_data(load_args args)`, multi-thread code snippet to load data. The function `load_threads` created by `pthread_create()` is defined in `./src/data.c:load_threads()`. There are total 32 threads by default.
11. `train` struct contains images base info such w, h, images address (pointer to pointer) in X, labels address in y et al. For each data batch, we always create 32 threads to fetch data, and train, and free `train` data struct.
12. `loss = train_network(net,train)` calls `./src/network.c:train_network()`. in `train_network()`, `n` represents how many batches. Befor training, first call `./src/data.c:get_net_batch()` ( copy `X.valus` in `train` to `net-input` address row by row, copy `y.vals` in `train` to `net->truth`. Why we need copy operation? Because before a next new iteration, we will free the `train` data structure (`./examples/classifer.c:140`)). After everything done, train the network `./src/network.c:train_network_datum(net)`
13. In `./src/network.c:forward_network(net)`, call `layer::forward()` layer by layer. Parameters in `net` obj, `input` and `output` are address, `inputs` and `outputs` are elements in input and output data.
14. `forward_convolutional_layer(convolutional_layer l, network net)`. `fill_cpu(...)`: fill `l.output` with 0 in a length of `l.outputs * l.batch`. In `convolutional_layer.c`, `l.n` is number of kernel, `l.m` is grouped (default l.groups=1) number of kernel. `k` is total elements in a kernel group, `l.size` is kernel size, `l.c` is kernel channels. `n` is the number of elements in a feature map, `l.out_w` and `l.out_h` is the width and hight of feature map. If `l.size == 1`, meaning don't need to execute conv. `*b` is put `im2col_cpu()` result. `l.h` and `l.w` is the heigh and width of input image. `l.stride`, is the kernel moving stride. `l.pad` is image padding size for conv.
15. `l.batch_normalize()` and `acrivate_array` are also in `forward_convolutional_layer()` function.
16. `calc_network_cost()` is in `forward_network()`. If we find a layer has cost, sum it up and compute average.
17. `backward_netword()` calls `l.backward(l,net)`, and the true backward function as last will be called for specific layers such as softmax_layer.c, avgpool_layer.c, et al.
18. `*net->seen` is the number of total trained images.
19. Save `*.backpu` and `*.weights` files if needed according some iteration setting. When train done, free all the resource, such as `net`, `labels`, `plist` et al., and exist.

### validation Processes
```
$gdb ./darknet
(gdb) b main
(gdb) run classifier valid cfg/cifar.data cfg/cifar_small.cfg backup/cifar_small.backup
```
1. `darknet.c` -- call -->`./examples/classifier.c:run_classifier()` -- call -->`validate_classifier_single()`
2. crop image, and call `network_predict(net, crop.data)`.

### A Unreasonable Example for Playing
To check how each layer works, I make an unreasonable example include many specific layers.
```
$gdb ./darknet
(gdb) b train_classifier
(gdb) run classifier train cfg/cifar.data cfg/cifar_small_un.cfg 
```
1. The called point for each layer is at `./src/network.c:204`. For `connected` layer, call `./src/conneted_layer.c:forward_connected_layer()`.
