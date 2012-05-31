#!/usr/bin/python

import cPickle, string, numpy, getopt, sys, random, time, re, pprint
from termcolor import colored, cprint
from mdhoffma import onlineldavb
from mdhoffma import wikirandom

def top10(lambdak):
  t = [(v, i) for (i, v) in enumerate(lambdak) if v != 0]
  t = sorted(t, reverse = True)[:10]
  return [vocab[v] for (_, v) in t]

def printtopics(newlambda, oldlambda):
  lines = []
  for k in range(0, len(newlambda)):
    oldtop10 = top10(prevlambda[k,:])
    newtop10 = top10(newlambda[k,:])

    s = colored(k, 'green') + ' '
    for i, w in enumerate(newtop10):
      try:
        oldi = oldtop10.index(w)
        s += w
        if oldi > i:
          s += colored('+' * (oldi - i), 'green')
        elif oldi < i:
          s += colored('-' * (i - oldi), 'red')
      except Exception, e:
        s += colored(w, 'green')
      s += ' '
    deleted  = set(oldtop10).difference(set(newtop10))
    s += ' ' + colored(' '.join(deleted), 'red')
    lines.append(s)
  print '\n'.join(lines)
    

# The number of documents to analyze each iteration
batchsize = 64
# The total number of documents in Wikipedia
D = 3.3e6
# The number of topics
K = 40

# Our vocabulary
vocab = file('mdhoffma/dictnostops.txt').read().split()
W = len(vocab)

# Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7

# olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
cprint('Loading model', 'yellow')
olda = cPickle.load(open('olda.pickle'))

prevlambda = olda._lambda

while True:
  it = olda._updatect
  cprint('Iteration %s: download %s random articles from Wikipedia' % (it, batchsize), 'yellow')
  (docset, articlenames) =  wikirandom.get_random_wikipedia_articles(batchsize)

  cprint('Give them to online LDA', 'yellow')
  olda.update_lambda(docset)
  
  printtopics(olda._lambda, prevlambda)
  prevlambda = olda._lambda
  
  cprint('Saving model', 'yellow')
  cPickle.dump(olda, open('olda.pickle', 'wb'))