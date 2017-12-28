import math
import dynet as dy
from scipy.stats import norm
from xnmt.batcher import *
from xnmt.serializer import *


class Attender(object):
  '''
  A template class for functions implementing attention.
  '''

  def __init__(self, input_dim):
    """
    :param input_dim: every attender needs an input_dim
    """
    pass

  def init_sent(self, sent):
    raise NotImplementedError('init_sent must be implemented for Attender subclasses')

  def calc_attention(self, state):
    raise NotImplementedError('calc_attention must be implemented for Attender subclasses')

class MlpAttender(Attender, Serializable):
  '''
  Implements the attention model of Bahdanau et. al (2014)
  '''

  yaml_tag = u'!MlpAttender'

  def __init__(self, yaml_context, input_dim=None, state_dim=None, hidden_dim=None):
    input_dim = input_dim or yaml_context.default_layer_dim
    state_dim = state_dim or yaml_context.default_layer_dim
    hidden_dim = hidden_dim or yaml_context.default_layer_dim
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    param_collection = yaml_context.dynet_param_collection.param_col
    self.pW = param_collection.add_parameters((hidden_dim, input_dim))
    self.pV = param_collection.add_parameters((hidden_dim, state_dim))
    self.pb = param_collection.add_parameters(hidden_dim)
    self.pU = param_collection.add_parameters((1, hidden_dim))
    self.curr_sent = None

  def init_sent(self, sent):
    self.attention_vecs = []
    self.curr_sent = sent
    I = self.curr_sent.as_tensor()
    W = dy.parameter(self.pW)
    b = dy.parameter(self.pb)

    self.WI = dy.affine_transform([b, W, I])
    wi_dim = self.WI.dim()
    # TODO(philip30): dynet affine transform bug, should be fixed upstream
    # if the input size is "1" then the last dimension will be dropped.
    if len(wi_dim[0]) == 1:
      self.WI = dy.reshape(self.WI, (wi_dim[0][0], 1), batch_size=wi_dim[1])

  def calc_attention(self, state):
    V = dy.parameter(self.pV)
    U = dy.parameter(self.pU)

    h = dy.tanh(dy.colwise_add(self.WI, V * state))
    scores = dy.transpose(U * h)
    if self.curr_sent.mask is not None:
      scores = self.curr_sent.mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return normalized

  def calc_context(self, state):
    attention = self.calc_attention(state)
    I = self.curr_sent.as_tensor()
    return I * attention

class DotAttender(Attender, Serializable):
  '''
  Implements dot product attention of https://arxiv.org/abs/1508.04025
  Also (optionally) perform scaling of https://arxiv.org/abs/1706.03762
  '''

  yaml_tag = u'!DotAttender'

  def __init__(self, yaml_context, scale=True):
    self.curr_sent = None
    self.attention_vecs = None
    self.scale = scale

  def init_sent(self, sent):
    self.curr_sent = sent
    self.attention_vecs = []
    self.I = dy.transpose(self.curr_sent.as_tensor())

  def calc_attention(self, state):
    scores = self.I * state
    if self.scale:
      scores /= math.sqrt(state.dim()[0][0])
    if self.curr_sent.mask is not None:
      scores = self.curr_sent.mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return normalized

  def calc_context(self, state):
    attention = self.calc_attention(state)
    I = self.curr_sent.as_tensor()
    return I * attention

class GeneralLinearAttender(Attender, Serializable):
  '''
  Implements the general linear attention of https://arxiv.org/abs/1508.04025
  '''

  yaml_tag = u'!GeneralLinearAttender'

  def __init__(self, yaml_context, input_dim=None, state_dim=None):
    input_dim = input_dim or yaml_context.default_layer_dim
    state_dim = state_dim or yaml_context.default_layer_dim
    self.input_dim = input_dim
    self.state_dim = state_dim
    param_collection = yaml_context.dynet_param_collection.param_col
    self.pWa = param_collection.add_parameters((input_dim, state_dim))
    self.curr_sent = None

  def init_sent(self, sent):
    self.curr_sent = sent
    self.attention_vecs = []
    self.I = self.curr_sent.as_tensor()

  def calc_attention(self, state):
    Wa = dy.parameter(self.pWa)
    scores = (dy.transpose(state) * Wa) * self.I
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return dy.transpose(normalized)

  def calc_context(self, state):
    attention = self.calc_attention(state)
    return self.I * attention

class BiasedAttender(Attender, Serializable):
  '''
  Interpolates between the distribution specified by any attender,
  and a decoding timestep-indexed specifiable prior 
  distribtion (currently only diagonal-normal.)
  '''

  yaml_tag = u'!BiasedAttender'

  def __init__(self, yaml_context, internal_attender, prior_type='normal', prior_args=None):
    self.internal_attender = internal_attender
    self.prior_type = prior_type
    self.prior_args = prior_args
    param_collection = yaml_context.dynet_param_collection.param_col
    self.pLamb = param_collection.add_parameters((1), init=dy.ConstInitializer(10))

  def init_sent(self, sent):
    self.internal_attender.init_sent(sent)
    self.generate_index = 0
    lamb = dy.parameter(self.pLamb)
    #print(lamb.value())
    #print("\n\nNEWSENT", len(self.internal_attender.curr_sent))

  def get_attention_bias(self, dec_index, source_length):
    if self.prior_type == 'normal':
      return dy.inputVector([norm.cdf(src_index - dec_index + .5) - norm.cdf(src_index - dec_index - .5)
          for src_index in range(source_length)])
    else:
      raise NotImplementedError("Only the 'normal' prior is implemented as of now.")

  def calc_attention(self, state, index=None):
    internal_attention = self.internal_attender.calc_attention(state)
    dec_index = index if index else self.generate_index
    attention_bias = self.get_attention_bias(dec_index, len(self.internal_attender.curr_sent))
    #print(self.generate_index)
    #print(attention_bias.dim())
    #print((10*attention_bias).value())
    #print(dy.transpose(internal_attention).dim())
    #print(dy.transpose(internal_attention).value())
    lamb = dy.parameter(self.pLamb)
    renormalized = dy.softmax((internal_attention) + dy.cmult(lamb, attention_bias))
    #print(dy.transpose(renormalized).dim())
    #print(dy.transpose(renormalized).value())
    #print()
    self.generate_index += 1
    return renormalized

  def calc_context(self, state):
    attention = self.calc_attention(state)
    return self.internal_attender.curr_sent.as_tensor() * attention

