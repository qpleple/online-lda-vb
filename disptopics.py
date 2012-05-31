import sys, os, re, random, math, urllib2, time, cPickle
import numpy
from termcolor import colored
import onlineldavb

def blah(testlambdaLine):
  lambdak = list(testlambdaLine)
  lambdak = lambdak / sum(lambdak)
  temp = zip(lambdak, range(0, len(lambdak)))
  return sorted(temp, key = lambda x: x[0], reverse=True)

def main():
    """
    Displays topics fit by onlineldavb.py. The first column gives the
    (expected) most prominent words in the topics, the second column
    gives their (expected) relative prominence.
    """
    print colored('Loading', 'yellow')
    vocab = str.split(file(sys.argv[1]).read())
    testlambda1 = numpy.loadtxt(sys.argv[2])
    testlambda2 = numpy.loadtxt(sys.argv[3])

    for k in range(0, len(testlambda1)):
        temp1 = blah(testlambda1[k, :])
        temp2 = blah(testlambda2[k, :])
        # print 'topic %d:' % (k)
        # feel free to change the "53" here to whatever fits your screen nicely.
        print colored("Topic %s" % k, 'green')
        print ' '.join([vocab[temp1[i][1]] for i in range(0, 10)])
        print colored(' '.join([vocab[temp2[i][1]] for i in range(0, 10)]), 'yellow')

if __name__ == '__main__':
    main()
