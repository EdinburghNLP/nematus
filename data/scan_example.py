import numpy
import theano

from theano import tensor


# some numbers
n_steps = 10
n_samples = 5
dim = 10
input_dim = 20
output_dim = 2


# one step function that will be used by scan
def oneStep(x_t, h_tm1, W_x, W_h, W_o):

    h_t = tensor.tanh(tensor.dot(x_t, W_x) +
                      tensor.dot(h_tm1, W_h))
    o_t = tensor.dot(h_t, W_o)

    return h_t, o_t

# spawn theano tensor variable, our symbolic input
# a 3D tensor (n_steps, n_samples, dim)
x = tensor.tensor3(dtype='float32')

# initial state of our rnn
init_state = tensor.alloc(0., n_samples, dim)

# create parameters that we will use,
# note that, parameters are theano shared variables

# parameters for input to hidden states
W_x_ = numpy.random.randn(input_dim, dim).astype('float32')
W_x = theano.shared(W_x_)

# parameters for hidden state transition
W_h_ = numpy.random.randn(dim, dim).astype('float32')
W_h = theano.shared(W_h_)

# parameters from hidden state to output
W_o_ = numpy.random.randn(dim, output_dim).astype('float32')
W_o = theano.shared(W_o_)

# scan function
([h_vals, o_vals], updates) = theano.scan(
    fn=oneStep,
    sequences=[x],
    outputs_info=[init_state, None],
    non_sequences=[W_x, W_h, W_o],
    n_steps=n_steps,
    strict=True)

# let us now compile a function to get the output
f = theano.function([x], [h_vals, o_vals])

# now we will call the compiled function with actual input
actual_input = numpy.random.randn(
    n_steps, n_samples, input_dim).astype('float32')
h_vals_, o_vals_ = f(actual_input)

# print the shapes
print 'shape of input :', actual_input.shape
print 'shape of h_vals:', h_vals_.shape
print 'shape of o_vals:', o_vals_.shape
