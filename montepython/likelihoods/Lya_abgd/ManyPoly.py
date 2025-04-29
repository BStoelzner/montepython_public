import copy
import numpy as np

#NUM_PLOTS = 500
polyType = np.polynomial.polynomial.Polynomial
def fitPoly(x,y,degree=2):
  #global NUM_PLOTS
  #print(NUM_PLOTS)
  beta = polyType.fit(x,y,degree)

  #import matplotlib.pyplot as plt
  #if NUM_PLOTS >= 0 and NUM_PLOTS%100==0:
  #  plt.figure()
  #  plt.plot(x,y)
  #  plt.plot(x,beta(x))
  #NUM_PLOTS-=1
  #if NUM_PLOTS == 0:
  #  plt.show()
  return beta

def fitManyPoly(x,y,xscale,degree=2):
  betas = [None for xval in x]
  for i,xval in enumerate(x):
    xmask = np.logical_and(x>x[i]-xscale,x<x[i]+xscale)
    betas[i] = fitPoly(x[xmask],y[xmask],degree=degree)
  return betas

class ManyPoly(object):
  
  def __init__(self,x,y,xscale,degree=2):
    assert(degree>0)
    self.x = x
    self.N = len(x)
    self.betas = fitManyPoly(x,y,xscale,degree=degree)

  @classmethod
  def construct_fast(objref, betas, x, N):
      self = object.__new__(objref)
      self.N = N
      self.betas = betas
      self.x = x
      return self

  def derivative(self,order=1):
    newbetas = [self.betas[i].deriv(order) for i in range(self.N)]
    return self.construct_fast(newbetas,self.x,self.N)

  def eval(self):
    retarr = np.empty(self.N)
    for i in range(self.N):
      retarr[i] = self.betas[i](self.x[i])
    return retarr

  def copy(self):
    return copy.deepcopy(self)
