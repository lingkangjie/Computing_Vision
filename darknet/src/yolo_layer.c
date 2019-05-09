/* Command at 2019-05-07, by lingkangjie */
#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

    l.n = n; // prior boxes number for a yolo layer, typical is 3
    l.total = total;
    l.batch = batch; // training batch size
    l.h = h; // the hight of input feature map, I call it as 'yolo cell'
    l.w = w; // the width of input feature map
    l.c = n*(classes + 4 + 1); // output 'channel' size, please see yolov3.md
    l.out_w = l.w; // the size of input and output are just the same.
    l.out_h = l.h;
    l.out_c = l.c; // output 'channel' size
    l.classes = classes; // VOC is 20 classes
    l.cost = calloc(1, sizeof(float)); // cost for yolo layer
    l.biases = calloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs; // the size of input and output are just the same.
    l.truths = 90*(4 + 1); // an image almost has 90 ground truths
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

/** \brief Get the bx, by, bw, bh, as paper describes
 *
 * @param x a pointer to yolo output
 * @param biases
 * @param n the n-th of anchor, e.g. n= 6, 7, 8
 * @param index the index of output datum slice
 * @param i i-th cell in column
 * @param j j-th cell in row
 * @param lw width of input conv feature map
 * @param lh hight of input conv feature map
 * @param w width of resized input image
 * @param h width of resized input image
 * @param strinde equals to lw*lh
 */
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

/** \brief Compute the loss for box
 *
 */
float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    // the diff between truth box and anchor box
    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    // the diff between current predition and previous predition
    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}


/** \brief  To compute the loss for class
 *
 * In Darknet, denoted t as truth, y as predict, define:
 * delta = -loss_derivative = - (y-t) = 1-y ( if t==1)
 * delta = -loss_derivative = - (y-t) = 0-y ( if t!=1)
 */
void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)
{
    int n;
    if (delta[index]){
        delta[index + stride*class] = 1 - output[index + stride*class];
        if(avg_cat) *avg_cat += output[index + stride*class];
        return;
    }
    for(n = 0; n < classes; ++n){
        delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
}

/** \brief Return a special kind of data index, such as tx,ty,confidence,class probability
 *
 * Since we need to activate data from nearest conv layer (e.g. layer 81, see yolov3.md),
 * for each different properties of yolo input feature map, we use a different type of activations.
 * If we call 13 x 13 x 75 as a tensor, then we total have l.batch tensors to yolo layer.
 * Here n=3, means for a yolo layer we have 3 anchors.
 * entry = 0 or 4, 0 means we get tx entry, 4 means we get confidence entry.
 *
 *  When l.batch=1, the l.output scheme:
 *       ________
 *      /       /|
 *     /-------| |
 *    /       /| |
 *   /-------| | /
 *  /       /| |/ <--- anchor3, depth=(4+l.classes+1), n=2
 * /-------/ | /  
 * |       | |/ <--- anchor 2, depth=(4+l.classes+1), n=1
 * |13 x 13| /
 * |l.w*l.h|/  <--- anchor 1, depth=(4+l.classes+1), n=0
 * /-------/
 *
 *  For an anchor area:
 *
 *     /-------| 
 *    /       /| 
 *   /-------| | 
 *  /       /| |
 * /-------/ | /  
 * |       | |/ <- confidence (1 number) and classes probability (20 numbers, 20 classes)
 * |13 x 13| /   - entry = 4
 * |l.w*l.h|/  <--- coordinate for an object, tx, ty, tw, th. (4 numbers)
 * /-------/     -- entry = 0
 *
 * The l.input and l.output have the same data size, but the meaning is different.
 *
 *
 */
static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h); // call from 178,182: location = n * l.w * l.h
    int loc = location % (l.w*l.h); // call from 178,182: loc = 0 always
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float)); // copy @param 2 to @param 1

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            /* why 2? Since here to activate tx and ty */
            /* TODO, why here leave out tw, th?*/
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4); // we get the index of confidence
            /* 1 + l.classes is the depth of confidence and class probability 
             * in feature map
             */
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float)); // initialize l.delta to 0
    if(!net.train) return;
    float avg_iou = 0; // average IOU
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0; // average confidences in an image, use to print only
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < l.max_boxes; ++t){
                        /* get the truth coordinates of an object. net.truth is the
                         * a start pointer to an area storaged all truth coordinates.
                         * max_boxex=90, meaning our yolo net only has ability to 
                         * deal with an image almost contains 90 objects.
                         * l.truths, total truth coordinates and class index. Since
                         * max_boxes=90, tx,ty,tw,th,c, l.truths=90*5=450
                         * 1, is stride. reference box.c
                         */
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        /* suppose we have only 1 object in an image, so
                         * [tx1, ty1, tw1, th1, c1, 0, 0, ..., all zeros]
                         * In other words, we only store truths, others are 0
                         */
                        if(!truth.x) break; // break out l.mat_boxes
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    /* get confidence in output datum for each (i, j)*/
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index]; // sum over all confidences
                    l.delta[obj_index] = 0 - l.output[obj_index]; // get negative
                    if (best_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0; // we only need the best_iou and its correspondent delta
                    }
                    if (best_iou > l.truth_thresh) { // bigger than threshold, l.truth_thresh always == 1, these piece of code may not be accessed? (TODO)
                        // confidence target = 1
                        l.delta[obj_index] = 1 - l.output[obj_index]; // get delta

                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];  // only used in yolo9000
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        // compute gradient of classes
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
                        // comput gradient of box
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                    }
                }
            }
        }
        // after we have compute an image in the current batch, we reach here
        
        /* Find the most IOU of anchors between the truth box.
         * That is to say, which anchor will responsible for prediction.
         */
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for(n = 0; n < l.total; ++n){ // l.total=9, 9 anchors
                box pred = {0};
                pred.w = l.biases[2*n]/net.w; // get even index
                pred.h = l.biases[2*n+1]/net.h; // get odd index
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            // return the index of value(best_n) in array(l.mask), check utils.c
            int mask_n = int_index(l.mask, best_n, l.n);
            if(mask_n >= 0){ // if we successfully get the 'responsible' anchor, we begin to calculate its box, class, predict loss
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);

                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                /* why delta is target - output? Since YOLO use (-gradient) */
                l.delta[obj_index] = 1 - l.output[obj_index];

                int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class = l.map[class];// only yolo9000.cfg use it
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}

void backward_yolo_layer(const layer l, network net)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1); // just copy l.delta to net.delta
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

