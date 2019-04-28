#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

/** \brief Make a connected layer.
 *
 * @param batch a batch, equal to net.batch.
 * @param inputs neural input number.
 * @param outputs neural output number, default is 1.
 * @param activation activation type.
 * @param batch_normalize whether to perform batch normalize.
 * @param adam TODO
 * @return A connected layer.
 */
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);

/** \brief Forward data inlude activating.
 *
 */
void forward_connected_layer(layer l, network net);
void backward_connected_layer(layer l, network net);
void update_connected_layer(layer l, update_args a);

#ifdef GPU
void forward_connected_layer_gpu(layer l, network net);
void backward_connected_layer_gpu(layer l, network net);
void update_connected_layer_gpu(layer l, update_args a);
void push_connected_layer(layer l);
void pull_connected_layer(layer l);
#endif

#endif

