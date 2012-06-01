#!/usr/bin/python

import cPickle, string, numpy, getopt, sys, random, time, re, pprint
from termcolor import colored, cprint
from mdhoffma import onlineldavb
from mdhoffma import wikirandom
import numpy as n

def topwords(lambdak, vocab, n = 10):
  t = [(v, i) for (i, v) in enumerate(lambdak) if v != 0]
  t = sorted(t, reverse = True)[:n]
  return [vocab[v] for (_, v) in t]

def diffList(newlist, oldlist):
  s = ''
  for i, w in enumerate(newlist):
    try:
      oldi = oldlist.index(w)
      s += w
      if oldi > i:
        s += colored('+' * (oldi - i), 'green')
      elif oldi < i:
        s += colored('-' * (i - oldi), 'red')
    except Exception, e:
      s += colored(w, 'green')
    s += ' '
  deleted  = set(oldlist).difference(set(newlist))
  s += ' ' + colored(' '.join(deleted), 'red')
  return s
  
def histtopics(tops, k, n = 10):
  cprint('Topic %s history' % k, 'yellow')
  print ' '.join(tops[0][k][:n])
  for i in range(1, len(tops)):
    s = colored(i, 'green') + ' '
    s += diffList(tops[i][k][:n], tops[i-1][k][:n])
    print s

# tops = cPickle.load(open('tops.pickle'))
# histtopics(tops, 0)
# sys.exit()

def printtopics(tops, n = 20):
  lines = []
  newwords = tops[len(tops) - 1]

  if len(tops) == 1:
    for k in range(0, len(newwords)):
      s = colored(k, 'green') + ' '
      s += ' '.join(newwords[k][:n])
      lines.append(s)
  else:
    oldwords = tops[len(tops) - 2]
    
    for k in range(0, len(newwords)):
      s = colored(k, 'green') + ' '
      s += diffList(newwords[k][:n], oldwords[k][:n])
      lines.append(s)
  
  print '\n'.join(lines)

def resettopics(olda, topics):
  for k in topics:
    cprint('Reset topic %s' % k, 'yellow')
    olda._lambda[k] = 1*n.random.gamma(100., 1./100., olda._W)
    olda._Elogbeta = onlineldavb.dirichlet_expectation(olda._lambda)
    olda._expElogbeta = n.exp(olda._Elogbeta)
    
if (len(sys.argv) < 2):
  cprint(' Running with variable in memory! ', 'white', 'on_red')
elif sys.argv[1] == 'new':
  # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
  olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
  tops = []
elif sys.argv[1] == 'load':
  cprint('Loading model', 'yellow')
  if len(sys.argv) == 2:
    olda = cPickle.load(open('olda.pickle'))
    tops = cPickle.load(open('tops.pickle'))
  else:
    olda = cPickle.load(open('%s-model.pickle' % sys.argv[2]))
    tops = cPickle.load(open('%s-tops.pickle' % sys.argv[2]))
else:
  cprint(' Invalid argument ', 'white', 'on_red')
  sys.exit()

def main(olda, tops, save = False):
  # The number of documents to analyze each iteration
  batchsize = 1024
  # The total number of documents in Wikipedia
  D = 3.3e6
  # The number of topics
  K = 60
  # Our vocabulary
  vocab = file('mdhoffma/dictnostops.txt').read().split()
  W = len(vocab)
  
  prevlambda = olda._lambda
  while True:
    it = olda._updatect
    cprint('Iteration %s: download %s random articles from Wikipedia' % (it, batchsize), 'yellow')
    # start_time_it = time.time()
    (docset, articlenames) =  wikirandom.get_random_wikipedia_articles(batchsize)
    # cprint('%.1f seconds' % (time.time() - start_time_it), 'blue')
  
    # start_time = time.time()
    cprint('Give them to online LDA', 'yellow')
    olda.update_lambda(docset)
    # cprint('%.1f seconds' % (time.time() - start_time), 'blue')
    
    top = [topwords(olda._lambda[k,:], vocab, 30) for k in range(0, len(olda._lambda))]
    tops.append(top)
    printtopics(tops)
  
    if save:
      cprint('Saving model...', 'yellow')
      if it % 10 == 0:
        cPickle.dump(olda, open('olda.pickle', 'wb'))
      cPickle.dump(tops, open('tops.pickle', 'wb'))
    # cprint('%.1f seconds for the iteration' % (time.time() - start_time_it), 'blue')