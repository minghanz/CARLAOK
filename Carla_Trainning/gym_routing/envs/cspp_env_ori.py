import math
import numpy as np
import networkx as nx
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from problem import Problem
MAX_ARRAY_LENGTH = 100
DATA = "/media/mSATA/UM/Upper routing simulation/SUMOdata/routingdataset.hdf5"
class CsppEnv(gym.Env):
    metadata = {'render.modes': []}
    def __init__(self):
        self.problem = Problem(DATA) # load all the graph examples
        self.prunedistance = 1 # the maximum level to jump when pruning
        self.pointer = 1
        self.ppointer = 0
        self.nodelist = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.levellist = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.r1list = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.float32)
        self.costlist = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.float32)
        self.path = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.action_space = None
        self.obervation_space = spaces.Box(low=0.0, high=2.0, shape=(self.problem.numnodes["graph1"], 2*self.problem.numnodes["graph1"]+4))#, dtype=np.float32)
        self.viewer = None
    def _draw_obs(self, curnode):
        actionopts = np.zeros((self.problem.numnodes["graph1"],), dtype=np.float32)
        for (index, value) in enumerate(self.nodelist[0:self.pointer]):
            vindex = self.problem.graphs["graph1"].nodes[value]["Index"]
            if self.levellist[index] >= self.levellist[0:self.pointer][-1]-self.prunedistance:
                # TODO check graph u,v
                actionopts[vindex] = 1
        nodestate = np.zeros((self.problem.numnodes["graph1"],3), dtype=np.float32)
        #print(self.problem.instance.dest)
        destidx = self.problem.graphs["graph1"].nodes[self.problem.instance.dest]["Index"]
        nodestate[destidx, 0] = 2.0
        nodestate[:, 1] = self.problem.instance.maxR1
        #TODO change nodestate cost layer
        #for index in range(self.problem.num_nodes):
        #    nodestate[index, 2] = self.problem.mincost[index]/self.problem.mincost[self.problem.dest]
        for (index, value) in enumerate(self.nodelist[0:self.pointer]):
            nindex = self.problem.graphs["graph1"].nodes[value]["Index"]
            if value != self.problem.instance.dest:
                nodestate[nindex, 0] = self.levellist[index]#/float(self.problem.maxlevel) #TODO change maxlevel
            else:
                nodestate[nindex, 0] += self.levellist[index]#/float(self.problem.maxlevel)
            nodestate[nindex, 1] = self.r1list[index]
            nodestate[nindex, 2] = self.costlist[index]
        for p in self.path[0:self.ppointer]:
            pindex = self.problem.graphs["graph1"].nodes[p]["Index"]
            nodestate[pindex, 0] = 1.0

        edgestate = self.edgestate
        obs = np.concatenate((nodestate, actionopts[:, np.newaxis], edgestate), axis=1)
        reward = 0.0
        if self.pointer == 0:
            reward = -1.0
        if curnode == self.problem.instance.dest:
            reward = 1.0 + np.exp(-np.abs(self.problem.instance.optimal-self.problem.instance.primalbound))
        #if self.levellist[index] == self.levellist[0:self.pointer][-1]:
        #    reward
        done = False
        if self.pointer == 0 or curnode == self.problem.instance.dest:
            done = True
        info = {"Choices":self.nodelist[:]}
        return obs, reward, done, info
    def _respond_to_action(self, action):
        #assert action in self.nodelist[0:self.pointer], "%r (%s) invalid"%(action, type(action))
        assert action in self.nodelist, "%r (%s) invalid"%(action, type(action))
        # get the last occurrence, this favors pulse over prune
        #index = np.where(self.nodelist == action)[0][-1]
        actindexlist = []
        for idx, act in enumerate(self.nodelist):
            if act == action:
                actindexlist.append(idx)
        index = actindexlist[-1]
        levelthres = self.levellist[index]
        #assert self.levellist[index] >= (self.levellist[0:self.pointer][-1]-self.prunedistance), "prune out of distance"
        assert self.levellist[index] >= (self.levellist[-1]-self.prunedistance), "prune out of distance"
        #self.pointer = index + 1
        for idx in range(index, len(self.levellist)):
            if self.levellist[idx] > levelthres:
                self.nodelist = self.nodelist[0:idx]
                self.levellist = self.levellist[0:idx]
                self.r1list = self.r1list[0:idx]
                self.costlist = self.costlist[0:idx]
                break
        self.pointer = len(self.nodelist)-1
        return (self.nodelist.pop(index),
                self.levellist.pop(index),
                self.r1list.pop(index),
                self.costlist.pop(index))
    def _pulse(self, curr1, curcost, curlevel, curnode):
        if curnode != self.problem.instance.dest:
            self._change_labels(curr1, curcost, curnode)
            #if self.problem.visited[curnode] == 0:
            assert self.problem.graphs["graph1"].node[curnode]['visited'] == 0 , "Current node is appears on the path, making path a loop"
            self.ppointer += 1
            #self.path[self.ppointer-1] = curnode
            self.path.append(curnode)
            self.problem.graphs["graph1"].node[curnode]['visited'] = 1
            for (u, v) in self.problem.graphs["graph1"].out_edges(curnode):
                newr1 = curr1 + self.problem.graphs["graph1"][u][v]['r1']
                newcost = curcost + self.problem.graphs["graph1"][u][v]['c']
                newlevel = curlevel + 1
                if (newr1 <= (self.problem.instance.maxR1-self.problem.instance.R1underbar[v])) and \
                    (newcost <= (self.problem.instance.primalbound-self.problem.instance.Cunderbar[v])) and \
                    (not self._check_labels(newr1, newcost, v)) and \
                    (self.problem.graphs["graph1"].node[v]['visited'] == 0):
                    # pulse
                    self.pointer += 1
                    #self.nodelist[self.pointer-1] = v
                    self.nodelist.append(v)
                    #self.levellist[self.pointer-1] = newlevel
                    self.levellist.append(newlevel)
                    #self.r1list[self.pointer-1] = newr1
                    self.r1list.append(newr1)
                    #self.costlist[self.pointer-1] = newcost
                    self.costlist.append(newcost)
                    assert(self.pointer == len(self.nodelist))
                elif (self.problem.graphs["graph1"].node[v]['visited'] == 1):
                    # update visited node on loop
                    self._change_labels(newr1, newcost, v)
        else:
            self.ppointer += 1
            #self.path[self.ppointer-1] = curnode
            self.path.append(curnode)
            if (curcost <= self.problem.instance.primalbound) and (curr1 <= self.problem.instance.maxR1):
                #self.problem.instance.finalpath = self.path[0:self.ppointer]
                self.problem.instance.finalpath = self.path

                assert(len(path)==self.ppointer)
                self.problem.instance.r1star = curr1
                self.problem.instance.primalbound = curcost
    def _change_labels(self, r1, c, v):
            if (c <= self.problem.graphs["graph1"].nodes[v]['c1']):
                self.problem.graphs["graph1"].nodes[v]['c1'] = c
                self.problem.graphs["graph1"].nodes[v]['r11'] = r1
            elif (r1 <= self.problem.graphs["graph1"].nodes[v]['r12']):
                self.problem.graphs["graph1"].nodes[v]['c2'] = c
                self.problem.graphs["graph1"].nodes[v]['r12'] = r1
            else:
                self.problem.graphs["graph1"].nodes[v]['c3'] = c
                self.problem.graphs["graph1"].nodes[v]['r13'] = r1
    def _check_labels(self, r1, c, v):
            if ((r1>self.problem.graphs["graph1"].nodes[v]['r11'] and \
                 c>self.problem.graphs["graph1"].nodes[v]['c1']) or \
                (r1>self.problem.graphs["graph1"].nodes[v]['r12'] and \
                 c>self.problem.graphs["graph1"].nodes[v]['c2']) or \
                (r1>self.problem.graphs["graph1"].nodes[v]['r13'] and \
                 c>self.problem.graphs["graph1"].nodes[v]['c3'])):
                return True
            return False
    def step(self, action):
        curnode, curlevel, curr1, curcost = self._respond_to_action(action)
        if self.ppointer >= curlevel:
            for i in range(curlevel-1, self.ppointer):
                self.problem.graphs["graph1"].node[self.path[i]]['visited'] = 0
                self.path.pop()
            self.ppointer = curlevel - 1
            assert(len(self.path) == self.ppointer)
        self._pulse(curr1, curcost, curlevel, curnode)
        return self._draw_obs(curnode)
    def reset(self, **kargs):
        self.pointer = 1
        self.ppointer = 0
        self.nodelist = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.levellist = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.r1list = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.float32)
        self.costlist = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.float32)
        self.path = []
        self.problem.reset(**kargs)
        #populate edgestate
        self.edgestate1 = np.zeros((self.problem.numnodes["graph1"], self.problem.numnodes["graph1"]), dtype=np.float32)
        self.edgestate2 = np.zeros((self.problem.numnodes["graph1"], self.problem.numnodes["graph1"]), dtype=np.float32)
        for (u, v) in self.problem.graphs["graph1"].edges:
            uindex = self.problem.graphs["graph1"].nodes[u]["Index"]
            vindex = self.problem.graphs["graph1"].nodes[v]["Index"]
            self.edgestate1[uindex, vindex] = self.problem.graphs["graph1"][u][v]['r1']
            self.edgestate2[uindex, vindex] = self.problem.graphs["graph1"][u][v]['c']
        self.edgestate = np.concatenate((self.edgestate1, self.edgestate2), axis=-1)
        #Reset the graph to initial point
        self.pointer = 1
        self.ppointer = 0
        #self.nodelist[self.pointer-1] = self.problem.instance.start
        #self.nodelist.append(self.problem.instance.start)
        #self.levellist[self.pointer-1] = 1
        #self.levellist.append(1)
        #self.r1list[self.pointer-1] = 0.0
        #self.r1list.append(0.0)
        #self.costlist[self.pointer-1] = 0.0
        #self.costlist.append(0.0)
        #curnode = self.nodelist[self.pointer-1]
        #curlevel = self.levellist[self.pointer-1]
        #curr1 = self.r1list[self.pointer-1]
        #curcost = self.costlist[self.pointer-1]
        curnode = self.problem.instance.start
        curlevel = 1
        curr1 = 0.0
        curcost = 0.0
        self.pointer -= 1
        self._pulse(curr1, curcost, curlevel, curnode)
        return self._draw_obs(curnode)
    def render(self, mode='human', close=False):
        if mode == 'human':
            screen_width = 600
            screen_height = 400
            #world_width = self.problem.xrange
            super(MyEnv, self).render(mode=mode)
        elif mode == 'rgb_array':
            super(MyEnv, self).render(mode=mode)
        else:
            super(MyEnv, self).render(mode=mode)
