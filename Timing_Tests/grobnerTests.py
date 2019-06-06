#Python code to run Grobner basis resolving set timing tests in parallel
#CHECK: is sympy available on the cluster?
#...see http://bficores.colorado.edu/biofrontiers-it/cluster-computing/fiji/creating-and-managing-virtual-environments-with-python3
#NOTE: time.clock() precise enough on these cpus? -> run until more than some number of seconds, divide by total runs

from hammingGrobner import check_resolving_grobner, brute_force_resolve
import multiprocessing as mp
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import pickle
import time
import os

def writeDict(d, outFile):
  with open(outFile, 'wb') as o:
    pickle.dump(d, o, protocol=pickle.HIGHEST_PROTOCOL)

def readDict(inFile):
  d = {}
  with open(inFile, 'rb') as f:
    d = pickle.load(f)
  return d

def readTSV(inFile):
  L = []
  with open(inFile, 'r') as f:
    f.readline() #header
    for line in f:
      l = line.strip().split('\t')
      k = int(l[0])
      a = int(l[1])
      R = l[2].split(',')
      isResolving = l[3]=='True'
      L.append((k, a, R, isResolving))
  return L

def readFiles(prefix=''):
  data = {}
  if len(glob.glob(prefix+'brute_force_times_all.dict'))>0: data = mergeDicts(data, readDict(prefix+'brute_force_times_all.dict'))
  if len(glob.glob(prefix+'grobner_times_all.dict'))>0: data = mergeDicts(data, readDict(prefix+'grobner_times_all.dict'))
  if len(glob.glob(prefix+'alternate_brute_force_times_all.dict'))>0: data = mergeDicts(data, readDict(prefix+'alternate_brute_force_times_all.dict'))
  if len(glob.glob(prefix+'parallel_grobner_times_all.dict'))>0: data = mergeDicts(data, readDict(prefix+'parallel_grobner_times_all.dict'))
  if len(data)>0: return data
  for f in [f for f in glob.glob(prefix+'brute_force_*.dict')+glob.glob(prefix+'grobner_*.dict')+glob.glob(prefix+'alternate_*.dict')+glob.glob(prefix+'parallel_grobner_*.dict') if 'all' not in f]:
    data = mergeDicts(data, readDict(f))
  return data

def runAll(examples, prefix='', intermediate='', repeats=1, procs=1):
  data = readFiles(prefix=prefix)
  data = {1:data.get(1, {}), 2:data.get(2, {}), 3:data.get(3, {}), 4:data.get(4, {})}
  jobs = [(a, k, R, isResolving, funcNum, intermediate) for a in examples for k in examples[a] for (R, isResolving) in examples[a][k] for funcNum in [1,2,3] for _ in range(repeats-len([1 for (r,_,_) in data[funcNum].get(a, {}).get(k, {}) if r==R]))]
  pool = mp.Pool(processes=procs)
  results = pool.map_async(runJob, jobs)
  results = results.get()
  pool.close()
  pool.join()

  jobs = [(a, k, R, isResolving) for a in examples for k in examples[a] for (R, isResolving) in examples[a][k] for _ in range(repeats-len([1 for (r,_,_) in data[4].get(a, {}).get(k, {}) if r==R]))]
  results += [runGrobnerPar(a, k, R, isResolving, intermediate=prefix+'intermediate_parallel_grobner_'+str(a)+'.tsv', procs=procs) for (a, k, R, isResolving) in jobs]

  for (funcNum, a, k, R, isResolving, elapsed) in results:
    if a not in data[funcNum]: data[funcNum][a] = {}
    if k not in data[funcNum][a]: data[funcNum][a][k] = []
    data[funcNum][a][k].append((R, isResolving, elapsed))

  writeDict({1:data[1]}, prefix+'brute_force_times_all.dict')
  writeDict({2:data[2]}, prefix+'grobner_times_all.dict')
  writeDict({3:data[3]}, prefix+'alternate_brute_force_times_all.dict')
  writeDict({4:data[4]}, prefix+'parallel_grobner_times_all.dict')

  if intermediate and len(glob.glob(prefix+'intermediate_*.tsv'))>0:
    for f in glob.glob(prefix+'intermediate_*.tsv'): os.remove(f)

def runSize(a, funcNum, examples, prefix='', intermediate='', repeats=1, procs=1):
  data = readFiles(prefix=prefix) #funcNum -> a -> k -> (R, isResolving, time)
  data = {funcNum:{a:data.get(funcNum, {a:{}}).get(a, {})}}
  jobs = [(a, k, R, isResolving, funcNum, intermediate) for k in examples[a] for (R, isResolving) in examples[a][k] for _ in range(repeats-len([1 for (r,_,_) in data[funcNum][a].get(k, {}) if r==R]))]

  results = []
  if funcNum==4: results = [runGrobnerPar(a, k, R, isResolving, intermediate=intermediate, procs=procs) for (a, k, R, isResolving, _, _) in jobs]
  else:
    pool = mp.Pool(processes=procs)
    results = pool.map_async(runJob, jobs)
    results = results.get()
    pool.close()
    pool.join()

  for (funcNum, a, k, R, isResolving, elapsed) in results:
    if k not in data[funcNum][a]: data[funcNum][a][k] = []
    data[funcNum][a][k].append((R, isResolving, elapsed))

  outFile = ('brute_force' if funcNum==1 else ('grobner' if funcNum==2 else ('alternate_brute_force' if funcNum==3 else 'parallel_grobner')))+'_times_'+str(a)+'.dict'
  writeDict({funcNum:data[funcNum]}, prefix+outFile)

  if intermediate and len(glob.glob(prefix+'intermediate_'+('brute_force' if funcNum==1 else ('grobner' if funcNum==2 else ('alternate_brute_force' if funcNum==3 else 'parallel_grobner')))+'_'+str(a)+'.tsv'))>0:
    os.remove(prefix+'intermediate_'+('brute_force' if funcNum==1 else ('grobner' if funcNum==2 else ('alternate_brute_force' if funcNum==3 else 'parallel_grobner')))+'_'+str(a)+'.tsv')

def runJob(arg): #((a, k, R, isResolving, funcNum)):
  (a, k, R, isResolving, funcNum, intermediate) = arg
  print('run', 'a', a, 'k', k, 'func', funcNum)
  res = -1
  numRuns = 0
  start = time.clock()
  while (time.clock()-start)<2.: #run for at least 2 seconds
    res = checkResolvingHamming(R, k, map(str, range(a))) if funcNum==1 else (check_resolving_grobner(R, k, a) if funcNum==2 else brute_force_resolve(R, k, a))
    numRuns += 1
  elapsed = (time.clock() - start) / float(numRuns)
  if res!=isResolving:
    print('res != isResolving')
    elapsed = 'FAIL'
  if intermediate:
    with open(intermediate, 'a+') as o:
      o.write('\t'.join(map(str, [funcNum, a, k, ','.join(R), isResolving, elapsed]))+'\n')
  return (funcNum, a, k, R, isResolving, elapsed)

def runGrobnerPar(a, k, R, isResolving, intermediate='', procs=1):
  print('run', 'a', a, 'k', k, 'parallel grobner procs', procs)
  res = -1
  numRuns = 0
  start = time.clock()
  while (time.clock()-start)<2.:
    res = check_resolving_grobner(R, k, a, procs=procs)
    numRuns += 1
  elapsed = (time.clock() - start) / float(numRuns)
  if res!=isResolving:
    print('res != isResolving')
    elapsed = 'FAIL'
  if intermediate:
    with open(intermediate, 'a+') as o:
      o.write('\t'.join(map(str, [4, a, k, ','.join(R), isResolving, elapsed]))+'\n')
  return (4, a, k, R, isResolving, elapsed)

def hammingDist(a, b):
  return sum(1 for (x,y) in zip(a,b) if x!=y)

def checkResolvingHamming(R, k, alphabet):
  tags = {}
  for seq in product(alphabet, repeat=k):
    tag = ';'.join(map(str, [hammingDist(seq, r) for r in R]))
    if tag in tags: return False
    tags[tag] = 1
  return True

def combineFiles(prefix=''):
  data = {}
  for f in [f for f in glob.glob(prefix+'brute_force_*.dict')+glob.glob(prefix+'grobner_*.dict')+glob.glob(prefix+'alternate_*.dict')+glob.glob(prefix+'parallel_grobner_*.dict') if '_all' not in f]:
    b = readDict(f)
    data = mergeDicts(data, b)

  #only when the *_all.dict was not updated at the end... how to check this? delete intermediate files when run completes
  for f in glob.glob(prefix+'intermediate_*'):
    with open(f, 'r') as g:
      for line in g:
        l = line.strip().split('\t')
        funcNum = int(l[0])
        a = int(l[1])
        k = int(l[2])
        R = l[3].split(',')
        isResolving = (l[4]=='True')
        elapsed = float(l[5])
        if funcNum not in data: data[funcNum] = {}
        if a not in data[funcNum]: data[funcNum][a] = {}
        if k not in data[funcNum][a]: data[funcNum][a][k] = []
        data[funcNum][a][k].append((R, isResolving, elapsed))
    os.remove(f)

  if 1 in data: writeDict({1:data[1]}, prefix+'brute_force_times_all.dict')
  if 2 in data: writeDict({2:data[2]}, prefix+'grobner_times_all.dict')
  if 3 in data: writeDict({3:data[3]}, prefix+'alternate_brute_force_times_all.dict')
  if 4 in data: writeDict({4:data[4]}, prefix+'parallel_grobner_times_all.dict')

def mergeDicts(x,y):
  for funcNum in y:
    if funcNum not in x: x[funcNum] = {}
    for a in y[funcNum]:
      if a not in x[funcNum]: x[funcNum][a] = {}
      for k in y[funcNum][a]:
        if k not in x[funcNum][a]: x[funcNum][a][k] = []
        x[funcNum][a][k] += y[funcNum][a][k]
  return x

#show how many repeats have been collected
def dictTests(examples, repeats, maxSize=-1, prefix=''):
  data = readFiles(prefix=prefix)

  funcNames = {1:'Brute Force', 2:'Grobner', 3:'Alternate Brute Force', 4:'Parallel Grobner'}
  for funcNum in sorted(data.keys()):
    print(str(funcNum)+': '+funcNames[funcNum])
    for a in sorted(data[funcNum].keys()):
      for k in sorted(data[funcNum][a].keys()):
        if a*k<=maxSize:
          (res, nonRes) = (0, 0)
          for (R, isResolving, t) in data[funcNum][a][k]:
            if isResolving: res += 1
            else: nonRes += 1
          exRes = len(list(filter(lambda x: x[1]==True, examples[a][k])))*repeats
          exNonRes = len(list(filter(lambda x: x[1]==False, examples[a][k])))*repeats
        print('   a: '+str(a)+', k: '+str(k)+', size: '+str(a*k)+', resolving: '+str(res)+'/'+str(exRes)+', non-resolving: '+str(nonRes)+'/'+str(exNonRes)+', total: '+str(res+nonRes)+'/'+str(exRes+exNonRes)+('*****' if (res+nonRes!=exRes+exNonRes) else ''))

#############
### PLOTS ###
#############
#
def makePlots(prefix='', maxSize=25):
  data = readFiles(prefix=prefix) #funcNum -> a -> k -> (R, isResolving, time)
  for i in range(3): sizeVSTime(data[1], data[2], data[3], resType=i, prefix=prefix, maxSize=maxSize)
  for i in range(3):
    kVSTime(data[1], resType=i, title='Brute Force', prefix=prefix, maxSize=maxSize)
    kVSTime(data[2], resType=i, title='Grobner Basis', prefix=prefix, maxSize=maxSize)
    kVSTime(data[3], resType=i, title='Alternate Brute Force', prefix=prefix, maxSize=maxSize)
#  for i in range(3):
#    timeHists(data[1], resType=i, title='Brute Force', prefix=prefix, maxSize=maxSize)
#    timeHists(data[2], resType=i, title='Grobner Basis', prefix=prefix, maxSize=maxSize)
#    timeHists(data[3], resType=i, title='Alternate Brute Force', prefix=prefix, maxSize=maxSize)
  for i in range(3): timeHists(data[4], resType=i, title='Parallel Grobner', prefix=prefix, maxSize=maxSize)
  #size vs time for grobner and brute ... 3 plots: res, non res, all
  #k vs time, each a a different curve, grobner and brute, res, non res, all

def sizeVSTime(brute, grobner, alt, resType=0, prefix='', maxSize=25): #0 res, 1 non res, 2 all
  maxY = 0
  sizes = {}
  for a in brute:
    for k in brute[a]:
      s = int(np.power(a, k))
      if a*k>maxSize: break
      if s not in sizes: sizes[s] = []
      for (R, res, t) in brute[a][k]:
        if resType==0 and res==True: sizes[s].append(t)
        elif resType==1 and res==False: sizes[s].append(t)
        elif resType==2: sizes[s].append(t)
  X = sorted(sizes.keys())
  Y = [np.mean(sizes[s]) for s in X]
  E = [np.std(sizes[s]) for s in X]
  (X,Y,E) = zip(*[(x,y,e) for (x,y,e) in zip(X,Y,E) if not np.isnan(y)])
  maxY = max(Y) #sum(max(zip(Y,E), key=lambda x: x[0]))
  plt.errorbar(X, Y, yerr=E, label='brute', capsize=2, fmt='bo-')
#  plt.plot(X, Y, 'bo-', label='brute', markersize=5)

  sizes = {}
  for a in grobner:
    for k in grobner[a]:
      s = int(np.power(a, k))
      if a*k>maxSize: break
      if s not in sizes: sizes[s] = []
      for (R, res, t) in grobner[a][k]:
        if resType==0 and res==True: sizes[s].append(t)
        elif resType==1 and res==False: sizes[s].append(t)
        elif resType==2: sizes[s].append(t)
  X = sorted(sizes.keys())
  Y = [np.mean(sizes[s]) for s in X]
  E = [np.std(sizes[s]) for s in X]
  (X,Y,E) = zip(*[(x,y,e) for (x,y,e) in zip(X,Y,E) if not np.isnan(y)])
  maxY = maxY if maxY > max(Y) else max(Y) #maxY if maxY > sum(max(zip(Y,E), key=lambda x: x[0])) else sum(max(zip(Y,E), key=lambda x: x[0]))
  plt.errorbar(X, Y, yerr=E, label='grobner', capsize=2, fmt='ro-', markersize=5)
#  plt.plot(X, Y, 'ro-', label='grobner', markersize=5)

  sizes = {}
  for a in alt:
    for k in alt[a]:
      s = int(np.power(a, k))
      if a*k>maxSize: break
      if s not in sizes: sizes[s] = []
      for (R, res, t) in alt[a][k]:
        if resType==0 and res==True: sizes[s].append(t)
        elif resType==1 and res==False: sizes[s].append(t)
        elif resType==2: sizes[s].append(t)
  X = sorted(sizes.keys())
  Y = [np.mean(sizes[s]) for s in X]
  E = [np.std(sizes[s]) for s in X]
  (X,Y,E) = zip(*[(x,y,e) for (x,y,e) in zip(X,Y,E) if not np.isnan(y)])
  maxY = maxY if maxY > max(Y) else max(Y) #maxY if maxY > sum(max(zip(Y,E), key=lambda x: x[0])) else sum(max(zip(Y,E), key=lambda x: x[0]))
  plt.errorbar(X, Y, yerr=E, label='alt', capsize=2, fmt='go-', markersize=5)
#  plt.plot(X, Y, 'go-', label='alt', markersize=5)

#  plt.xlim([0, 7000])
  plt.ylim([-0.001, 1.01*maxY])
  plt.xlabel('Nodes')
  plt.ylabel('Time (sec)')
  plt.title('Resolving Sets' if resType==0 else ('Non-resolving Sets' if resType==1 else 'All Sets'))
  plt.legend(loc='upper left')
#  plt.show()
  plt.savefig(prefix+'size_'+str(resType)+'.png', bbox_inches='tight')
  plt.clf()

def kVSTime(info, resType=0, title='', prefix='', maxSize=25): #0 res, 1 non res, 2 all
  ks = {}
  for a in info:
    if a not in ks: ks[a] = {}
    for k in info[a]:
      if a*k>maxSize: break
      if k not in ks[a]: ks[a][k] = []
      for (R, res, t) in info[a][k]:
        if resType==0 and res==True: ks[a][k].append(t)
        elif resType==1 and res==False: ks[a][k].append(t)
        elif resType==2: ks[a][k].append(t)
  maxY = 0
  for a in sorted(ks.keys()):
    X = sorted(ks[a].keys())
    Y = [np.mean(ks[a][k]) for k in X]
    E = [np.std(ks[a][k]) for k in X]
    maxY = maxY if maxY > max(Y) else max(Y) #maxY if maxY > sum(max(zip(Y,E), key=lambda x: x[0])) else sum(max(zip(Y,E), key=lambda x: x[0]))
    plt.errorbar(X, Y, yerr=E, label=str(a), capsize=2, markersize=5)
    #plt.plot(X, Y, 'o-', label=str(a), markersize=5)
#  plt.xlim([-0.5, 12.5])
  #plt.ylim([-0.001, 1.01*maxY])
  plt.xlabel('k')
  plt.ylabel('Time (sec)')
  plt.title(title+'('+('Resolving Sets' if resType==0 else ('Non-resolving Sets' if resType==1 else 'All Sets'))+')')
  plt.legend(loc='upper left')
#  plt.show()
  name = prefix+title.replace(' ', '_')+'_'+str(resType)+'.png'
  plt.savefig(name, bbox_inches='tight')
  plt.clf()

def timeHists(info, resType=0, title='', prefix='', maxSize=25): #0 res, 1 non res, 2 all
  for a in info:
    for k in info[a]:
      if a*k>maxSize: break
      times = []
      for (R, res, t) in info[a][k]:
        if resType==0 and res==True: times.append(t)
        elif resType==1 and res==False: times.append(t)
        elif resType==2: times.append(t)
      plt.hist(times)
      plt.xlabel('Time (sec)')
      plt.ylabel('Count')
      plt.title(title+r' $a='+str(a)+', k='+str(k)+'$ '+'('+('Resolving Sets' if resType==0 else ('Non-resolving Sets' if resType==1 else 'All Sets'))+')')
#      plt.show()
      name = title.replace(' ','_')+'_'+str(a)+'_'+str(k)+'_hist.png'
      plt.savefig(name, bbox_inches='tight')
      plt.clf()

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Check Resolvability Timing')
  parser.add_argument('--a', type=int, default=2, required=False,
                      help='alphabet size (assumed to be <= 10')
  parser.add_argument('--funcNum', type=int, default=1, required=False,
                      help='method to test. (1) for brute force (2) for grobner basis approach (3) for alternate brute force (4) for parallel Grobner')
  parser.add_argument('--data', type=int, default=1, required=False,
                      help='which data set to use (1) res_set_data.tsv and res_set_data_2.tsv (2) res_set_data_3.tsv')
  parser.add_argument('--procs', type=int, default=1, required=False,
                      help='number of processes to use')
  parser.add_argument('--repeats', type=int, default=1, required=False,
                      help='number of times to run each example')
  parser.add_argument('--all', action='store_true',
                      help='flag if set runs all combinations in a single job')
  parser.add_argument('--combine', action='store_true',
                      help='flag if set combine files')
  parser.add_argument('--plot', action='store_true',
                      help='flag if set make simple plots')
  parser.add_argument('--test', action='store_true',
                      help='flag if set tet number of repeats recorded in full dicts')
  parser.add_argument('--maxSize', type=int, default=25, required=False,
                      help='the maximum a*k values to examine from the given data set')
  parser.add_argument('--intermediate', action='store_true',
                      help='flag if set save runs of each process. note that some writes on linux on atomic (not guaranteed here)')
  args = parser.parse_args()

  if args.a>10:
    print('a value too large, must be <=10. a:'+str(args.a))
    exit(0)

  print('READ TSV FILES')
  files = ['res_set_data_2.tsv', 'res_set_data.tsv'] if args.data==1 else ['res_set_data_3.tsv']
  prefix = 'orig_' if args.data==1 else ''
  intermediate = ''
  if args.intermediate: intermediate = prefix+'intermediate_'+('brute_force' if args.funcNum==1 else ('grobner' if args.funcNum==2 else ('alternate_brute_force' if args.funcNum==3 else 'parallel_grobner')))+'_'+str(args.a)+'.tsv'
  examples = {}
  for f in files:
    L = readTSV(f)
    for (k, a, R, isResolving) in L:
      if a*k<=args.maxSize: ### how variable are multiple runs on the same graph?
        if a not in examples: examples[a] = {}
        if k not in examples[a]: examples[a][k] = []
        examples[a][k].append((R, isResolving))

  print('START RUNS')
  if args.plot: makePlots(prefix=prefix, maxSize=args.maxSize)
  elif args.combine:
    combineFiles(prefix='orig_')
    combineFiles(prefix='')
  elif args.test: dictTests(examples, args.repeats, maxSize=args.maxSize, prefix=prefix)
  elif args.all: runAll(examples, prefix=prefix, intermediate=intermediate, repeats=args.repeats, procs=args.procs)
  else: runSize(args.a, args.funcNum, examples, prefix=prefix, intermediate=intermediate, repeats=args.repeats, procs=args.procs)

  print('DONE')
