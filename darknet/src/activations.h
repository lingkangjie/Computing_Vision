#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "cuda.h"
#include "math.h"

/** \brief Given a activation name, return a defined activation type.
 *
 * e.g., given 'tanh', return a enumeration type 'TANH'. If the given string
 * is not defined, return the default type 'RELU'.
 */ 
ACTIVATION get_activation(char *s);

/** \brief Given actication string, return its type.
 *
 * e.g, given 'LOGISTIC', return 'logistic'. If the given type is not defined,
 * return 'relu'.
 */
char *get_activation_string(ACTIVATION a);

/** \brief Given a data element and its activation type, return its activation.
 */
float activate(float x, ACTIVATION a);

/** \brief Given x, return its gradient.
 *
 * @param x an input data element to compute gradient.
 * @param a activation type.
 */
float gradient(float x, ACTIVATION a);

/** \brief Given an array of x, return all of its gradient multiply its delta.
 *
 * In mathematics, a neuron recive delta, multiply its activation gradient, and 
 * push the new delta backprop. In some activation function, the gradient is only 
 * related with activation output, but some related with activation input x.
 * @param x a const input array, the current layer output, such as feature map.
 * @param n total number of x to be compute, equal to l.batch * l.out_c * .out_w * l.out_h.
 * @param a activation type.
 * @param delta error term in BP.
 */
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);

/** \brief Given an input array x, compute all of its activations
 *
 * @param x a pointer to an input array, not const
 * @param n total number of x to be compute.
 * @param a activation type.
 */
void activate_array(float *x, const int n, const ACTIVATION a);
#ifdef GPU
void activate_array_gpu(float *x, int n, ACTIVATION a);
void gradient_array_gpu(float *x, int n, ACTIVATION a, float *delta);
#endif
// concrete activation function defined.
static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate(float x){return x;}
static inline float logistic_activate(float x){return 1./(1. + exp(-x));}
static inline float loggy_activate(float x){return 2./(1. + exp(-x)) - 1;}
static inline float relu_activate(float x){return x*(x>0);}
static inline float elu_activate(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
static inline float selu_activate(float x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x)-1);}
static inline float relie_activate(float x){return (x>0) ? x : .01*x;}
static inline float ramp_activate(float x){return x*(x>0)+.1*x;}
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}
static inline float tanh_activate(float x){return (exp(2*x)-1)/(exp(2*x)+1);}
static inline float plse_activate(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

// concrete gradient function defined
static inline float lhtan_activate(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}
static inline float lhtan_gradient(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

static inline float hardtan_gradient(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
static inline float linear_gradient(float x){return 1;}
static inline float logistic_gradient(float x){return (1-x)*x;}
static inline float loggy_gradient(float x)
{
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}
static inline float stair_gradient(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}
static inline float relu_gradient(float x){return (x>0);}
static inline float elu_gradient(float x){return (x >= 0) + (x < 0)*(x + 1);}
static inline float selu_gradient(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
static inline float relie_gradient(float x){return (x>0) ? 1 : .01;}
static inline float ramp_gradient(float x){return (x>0)+.1;}
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}
static inline float tanh_gradient(float x){return 1-x*x;}
static inline float plse_gradient(float x){return (x < 0 || x > 1) ? .01 : .125;}

#endif

