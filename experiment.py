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

def savetopics(olda):
  olda._Elogbeta = onlineldavb.dirichlet_expectation(olda._lambda)
  olda._expElogbeta = n.exp(olda._Elogbeta)
  

def mergetopics(olda, k1, k2):
  cprint('Merging both topics into %s' % k1, 'yellow')
  olda._lambda[k1] += olda._lambda[k2] * sum(olda._lambda[k1]) / sum(olda._lambda[k2])
  
  cprint('Reset topic %s' % k2, 'yellow')
  olda._lambda[k2] = 1*n.random.gamma(100., 1./100., olda._W)
  savetopics(olda)

def resettopics(olda, topics):
  for k in topics:
    cprint('Reset topic %s' % k, 'yellow')
    olda._lambda[k] = 1*n.random.gamma(100., 1./100., olda._W)
    savetopics(olda)
    
if (len(sys.argv) < 2):
  pass
elif sys.argv[1] == 'new':
  # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
  olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
  tops = []
elif sys.argv[1] == 'load':
  if len(sys.argv) == 2:
    cprint(' What model? ', 'white', 'on_red')
    sys.exit()
    # olda = cPickle.load(open('olda.pickle'))
    # tops = cPickle.load(open('tops.pickle'))
  else:
    cprint('Loading model %s' % sys.argv[2], 'yellow')
    olda = cPickle.load(open('%s-model.pickle' % sys.argv[2]))
    tops = cPickle.load(open('%s-tops.pickle' % sys.argv[2]))
else:
  cprint(' Invalid argument ', 'white', 'on_red')
  sys.exit()


merging = []
def main(olda, tops, n = 20):
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
  lastmerge = 0
  while True:
    it = olda._updatect
    cprint('Iteration %s: download %s random articles from Wikipedia' % (it, batchsize), 'yellow')
    (docset, articlenames) =  wikirandom.get_random_wikipedia_articles(batchsize)
  
    cprint('Give them to online LDA', 'yellow')
    olda.update_lambda(docset)
    
    top = [topwords(olda._lambda[k,:], vocab, 30) for k in range(0, len(olda._lambda))]
    tops.append(top)
    printtopics(tops, n)

    cprint('Top 3 correlated topics', 'yellow')
    corr = topcorrelated(olda)
    for i in range(0, 3):
      (dot, (k1, k2)) = corr[i]
      cprint('Correlation %s' % dot, 'green')
      s = colored(k1, 'green')
      s += ' ' + ' '.join(topwords(olda._lambda[k1], vocab, n))
      print s
      s = colored(k2, 'green')
      s += ' ' + ' '.join(topwords(olda._lambda[k2], vocab, n))
      print s
    
    
    (dot, (k1, k2)) = corr[0]
    if dot > 0.5 and it - lastmerge > 10:
      cprint(' Merging topics %s and %s ' % (k1, k2), 'white', 'on_blue')
      mergetopics(olda, k1, k2)
      merging.append(corr[0])
      cPickle.dump(merging, open('merging', 'wb'))
      lastmerge = it

def norms(olda, t):
  l = olda._lambda[t]
  values = [(n.dot(l, lk), k) for (k, lk) in enumerate(list(olda._lambda)) if k != t]
  return sorted(values)

def maxdot(olda):
  l = olda._lambda
  mdot = 0
  midx = (0, 0)
  for i in range(0, len(l) - 1):
    for j in range(i + 1, len(l)):
      dot = n.dot(l[i], l[j])
      if dot > mdot:
        mdot = dot
        midx = (i, j)
        print 'new max: %s for the topics %s-%s' % (colored(mdot, 'yellow'), colored(i, 'yellow'), colored(j, 'yellow'))
  return midx

def topcorrelated(olda):
  l = olda._lambda
  dots = []
  for i in range(0, len(l) - 1):
    for j in range(i + 1, len(l)):
      dots.append((n.dot(l[i] / n.linalg.norm(l[i]), l[j] / n.linalg.norm(l[j])), (i, j)))
  return sorted(dots, reverse = True)

def propmerge(olda):
  vocab = file('mdhoffma/dictnostops.txt').read().split()
  corr  = topcorrelated(olda)
  corr  = [(dot, idx) for (dot, idx) in corr if dot > 0.5]
  for (dot, idx) in corr:
    cprint('Correlation %s' % dot, 'yellow')
    (k1, k2) = idx

    s = colored(k1, 'green')
    s += ' ' + ' '.join(topwords(olda._lambda[k1], vocab, 20))
    print s

    s = colored(k2, 'green')
    s += ' ' + ' '.join(topwords(olda._lambda[k2], vocab, 20))
    print s
