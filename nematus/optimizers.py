'''
Optimizers
'''

import numpy
from collections import OrderedDict

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from util import *
from theano_util import *

# Calling convention:
# f_update = name(hyperp, tparams, grads, inputs (list), cost)
# with profile as an optional argument

def adam(lr, tparams, grads, inp, cost, beta1=0.9, beta2=0.999, e=1e-8, optimizer_params={}, profile=False):
    PREFIX='adam_'

    updates = []
    optimizer_tparams = {}

    # Avoid underflow of e with float16
    if (floatX == "float16") and (e > 0.0):
        e = max(e, 1e-6)

    t_prev_name = PREFIX + 't_prev'
    if t_prev_name in optimizer_params:
        t_prev_init = optimizer_params[t_prev_name]
    else:
        t_prev_init = 0.
    t_prev = theano.shared(numpy_floatX(t_prev_init), t_prev_name)
    optimizer_tparams[t_prev_name] = t_prev
    
    t = t_prev + 1.
    lr_t = lr * tensor.sqrt(1. - beta2**t) / (1. - beta1**t)

    for p, g in zip(tparams.values(), grads):
        # Create/Load variable for first moment
        m_name = PREFIX + p.name + '_mean'
        if m_name in optimizer_params:
            m_init = optimizer_params[m_name]
        else:
            m_init = p.get_value() * 0.
        m = theano.shared(m_init, m_name)
        optimizer_tparams[m_name] = m

        # Create/Load variable for second moment
        v_name = PREFIX + p.name + '_variance'
        if v_name in optimizer_params:
            v_init = optimizer_params[v_name]
        else:
            v_init = p.get_value() * 0.
        v = theano.shared(v_init, v_name)
        optimizer_tparams[v_name] = v

        # Define updates on shared vars
        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g**2
        step = lr_t * m_t / (tensor.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))

    f_update = theano.function([lr]+inp, cost, updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_update, optimizer_tparams

def adadelta(lr, tparams, grads, inp, cost, optimizer_params={}, profile=False):
    PREFIX = 'adadelta_'

    updates = []
    optimizer_tparams = {}

    for p, g in zip(tparams.values(), grads):
        zg_name = PREFIX + p.name + '_zg'
        if zg_name in optimizer_params:
            zg_init = optimizer_params[zg_name]
        else:
            zg_init = p.get_value() * 0.
        zg = theano.shared(zg_init, zg_name)
        optimizer_tparams[zg_name] = zg

        ru2_name = PREFIX + p.name + '_ru2'
        if ru2_name in optimizer_params:
            ru2_init = optimizer_params[ru2_name]
        else:
            ru2_init = p.get_value() * 0.
        ru2 = theano.shared(ru2_init, ru2_name)
        optimizer_tparams[ru2_name] = ru2 

        rg2_name = PREFIX + p.name + '_rg2'
        if rg2_name in optimizer_params:
            rg2_init = optimizer_params[rg2_name]
        else:
            rg2_init = p.get_value() * 0.
        rg2 = theano.shared(rg2_init, rg2_name)
        optimizer_tparams[rg2_name] = rg2 

        ud = -tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
        updates.append((zg, g))
        updates.append((rg2, 0.95 * rg2 + 0.05 * (g ** 2)))
        updates.append((ru2, 0.95 * ru2 + 0.05 * (ud ** 2)))
        updates.append((p, p + ud))

    f_update = theano.function([lr]+inp, cost, updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_update, optimizer_tparams

def rmsprop(lr, tparams, grads, inp, cost, optimizer_params={}, profile=False):
    PREFIX = 'rmsprop_'

    updates = []
    optimizer_tparams = {}

    for p, g in zip(tparams.values(), grads):
        zg_name = PREFIX + p.name + '_zg'
        if zg_name in optimizer_params:
            zg_init = optimizer_params[zg_name]
        else:
            zg_init = p.get_value() * 0.
        zg = theano.shared(zg_init, zg_name)
        optimizer_tparams[zg_name] = zg

        rg_name = PREFIX + p.name + '_rg'
        if rg_name in optimizer_params:
            rg_init = optimizer_params[rg_name]
        else:
            rg_init = p.get_value() * 0.
        rg = theano.shared(rg_init, rg_name)
        optimizer_tparams[rg_name] = rg

        rg2_name = PREFIX + p.name + '_rg2'
        if rg2_name in optimizer_params:
            rg2_init = optimizer_params[rg2_name]
        else:
            rg2_init = p.get_value() * 0.
        rg2 = theano.shared(rg2_init, rg2_name)
        optimizer_tparams[rg2_name] = rg2

        ud_name = PREFIX + p.name + '_ud'
        if ud_name in optimizer_params:
            ud_init = optimizer_params[ud_name]
        else:
            ud_init = p.get_value() * 0.
        ud = theano.shared(ud_init, ud_name)
        optimizer_tparams[ud_name] = ud 

        updates.append((zg, g))
        updates.append((rg, 0.95 * rg + 0.05 * g))
        updates.append((rg2, 0.95 * rg2 + 0.05 * (g ** 2)))

        udn = 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)
        updates.append((ud, udn))
        updates.append((p, p + udn))


    f_update = theano.function([lr]+inp, cost, updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_update, optimizer_tparams

def sgd(lr, tparams, grads, inp, cost, optimizer_params=None, profile=False):
    updates = [(p, p - lr * g) for p, g in zip(tparams.values(), grads)]
    f_update = theano.function([lr]+inp, cost, updates=updates, profile=profile)

    return f_update, {}

def sgdmomentum(lr, tparams, grads, inp, cost, momentum=0.9, optimizer_params={}, profile=False):
    assert momentum >= 0 and momentum < 1
    PREFIX = 'sgdmomentum_'

    updates = []
    optimizer_tparams = {}

    for p, g in zip(tparams.values(), grads):
        prev_name = PREFIX + p.name + '_prev'
        if prev_name in optimizer_params:
            prev_init = optimizer_params[prev_name]
        else:
            prev_init = p.get_value() * 0.
        prev = theano.shared(prev_init, prev_name)
        optimizer_tparams[prev_name] = prev
        step = momentum * prev - lr * g
        updates.append((prev, step))
        updates.append((p, p + step))

    f_update = theano.function([lr]+inp, cost, updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_update, optimizer_tparams
