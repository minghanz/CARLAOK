import math
import numpy as np
import networkx as nx
import gym
import h5py

class ProblemInstance:
    def __init__(self, route=None, traffic=None):
        self.route = route
        self.traffic = traffic
    def reset(self, graph):
        # initialize the graph with this problem instance
        self.primalbound = np.inf
        self.r1star = np.inf
        self.finalpath = []
        for node in graph.nodes:
            graph.nodes[node]['visited'] = 0

            graph.nodes[node]['r11'] = np.inf
            graph.nodes[node]['r12'] = np.inf
            graph.nodes[node]['r13'] = np.inf

            graph.nodes[node]['c1'] = np.inf
            graph.nodes[node]['c2'] = np.inf
            graph.nodes[node]['c3'] = np.inf

        for (idx, (u, v)) in enumerate(graph.edges):
            graph[u][v]['r1'] = self.traffic[idx,1] if self.traffic[idx,1] > 0 else -self.traffic[idx,1]
            graph[u][v]['c'] = self.traffic[idx,3] if self.traffic[idx,3] > 0 else 0.0
        self.start = int(self.route[1])
        self.dest = int(self.route[2])

        self.R1underbar = nx.shortest_path_length(graph, target=self.dest, weight='r1')
        self.Cunderbar = nx.shortest_path_length(graph, target=self.dest, weight='c')
        self.maxR1 = self.R1underbar[self.start]*1.2

        self.optimal = None
        for path in nx.shortest_simple_paths(graph, self.start, self.dest, weight='c'):
            cost = 0.0
            res1 = 0.0
            for (idx, n) in enumerate(path):
                if idx > 0:
                    cost += graph[path[idx-1]][n]['c']
                    res1 += graph[path[idx-1]][n]['r1']
            if res1 <= self.maxR1:
                self.optimal = cost
                break
class Problem:
    def __init__(self, filename, val_frac=0.7):
        self.val_frac = val_frac
        print("validation fraciton: {}".format(self.val_frac))

        print("loading data...")
        self._load_data(filename)

    def _load_data(self, filename):
        f = h5py.File(filename, 'r')
        ri = f['graph1/route_info']
        ti = f['graph1/traffic_info']
        gs = f['graph1/graph_structure']
        self.graphs = {"graph1":nx.DiGraph()}
        for link in gs:
            self.graphs["graph1"].add_edge(link[0], link[1])
        for idx, node in enumerate(self.graphs["graph1"].nodes):
            self.graphs["graph1"].nodes[node]["Index"] = idx
        self.numnodes = {"graph1":len(self.graphs["graph1"].nodes())}
        self.numsteps = ri.shape[ri.attrs["stepaxis"]-1]
        self.numroutes = ri.shape[ri.attrs["routeaxis"]-1]
        self.numproblems = self.numsteps*self.numroutes

        self.numsteptrain = int(math.floor(self.val_frac*self.numsteps))
        self.numroutetrain = int(math.floor(self.val_frac*self.numroutes))

        self.numstepval = self.numsteps - self.numsteptrain
        self.numrouteval = self.numroutes - self.numroutetrain

        self.numtrain = self.numroutetrain * self.numsteptrain
        self.numval = self.numproblems - self.numtrain

        self.numstudy1 = self.numrouteval * self.numsteptrain #seen step unseen routes
        self.numstudy2 = self.numroutetrain * self.numstepval #unseen step seen routes
        self.numstudy3 = self.numrouteval * self.numstepval # both unseen
        # shuffle data
        stepp = np.random.permutation(self.numsteps)
        routep = np.random.permutation(self.numroutes)
        print("Shift and scale")
        self._shift_scale(ti)

        self.train = []
        self.val = []
        self.study1 = []
        self.study2 = []
        self.study3 = []
        for i, step in enumerate(stepp):
            for j, rid in enumerate(routep):
                if i < self.numsteptrain and j < self.numroutetrain:
                    self.train.append(ProblemInstance(route=ri[step,rid,:],traffic=self.ti_scaled[:,step,:]))
                else:
                    self.val.append(ProblemInstance(route=ri[step,rid,:],traffic=self.ti_scaled[:,step,:]))

        for step in stepp[:self.numsteptrain]:
            for rid in routep[self.numroutetrain:]:
                self.study1.append(ProblemInstance(route=ri[step,rid,:],traffic=self.ti_scaled[:,step,:]))

        for step in stepp[self.numsteptrain:]:
            for rid in routep[:self.numroutetrain]:
                self.study2.append(ProblemInstance(route=ri[step,rid,:],traffic=self.ti_scaled[:,step,:]))

        for step in stepp[self.numsteptrain:]:
            for rid in routep[self.numroutetrain:]:
                self.study3.append(ProblemInstance(route=ri[step,rid,:],traffic=self.ti_scaled[:,step,:]))

        # reset ptrs
        self.reset_ptr_train()
        self.reset_ptr_val()
        self.reset_ptr_study1()
        self.reset_ptr_study2()
        self.reset_ptr_study3()


    def _shift_scale(self,ti):
        self.shiftfeatures = np.mean(ti, axis=(0,1))
        self.scalefeatures = np.std(ti, axis=(0,1))
        assert(self.shiftfeatures.shape[0]==ti.shape[ti.attrs["featureaxis"]-1])
        #self.ti_scaled = (ti[:,:,:] - self.shiftfeatures)/self.scalefeatures
        self.ti_scaled = ti[:,:,:]
    def reset(self, **karg):
        # this function should reset the problem instance
        if "train" in karg:
            return self.next_train()
        elif "val" in karg:
            return self.next_val()
        elif "study1" in karg:
            return self.next_study1()
        elif "study2" in karg:
            return self.next_study2()
        elif "study3" in karg:
            return self.next_study3()
        return self.next_train()
    # Sample new problem for validation
    def next_train(self):
        trainidx = self.permutation_train[self.ptr_train]
        # initialize the graph
        self.train[trainidx].reset(self.graphs["graph1"])
        self.instance = self.train[trainidx]
        self.ptr_train += 1
    def reset_ptr_train(self):
        self.permutation_train = np.random.permutation(self.numtrain)
        self.ptr_train = 0
    # Sample new problem for validation
    def next_val(self):
        validx = self.permutation_val[self.ptr_val]
        # initialize the graph
        self.val[validx].reset(self.graphs["graph1"])
        self.instance = self.val[validx]
        self.ptr_val += 1
    def reset_ptr_val(self):
        self.permutation_val = np.random.permutation(self.numval)
        self.ptr_val = 0
    # Sample new problem for study1
    def next_study1(self):
        study1idx = self.permutation_study1[self.ptr_study1]
        # initialize the graph
        self.study1[study1idx].reset(self.graphs["graph1"])
        self.instance = self.study1[study1idx]
        self.ptr_study1 += 1
    def reset_ptr_study1(self):
        self.permutation_study1 = np.random.permutation(self.numstudy1)
        self.ptr_study1 = 0
    # Sample new problem for study2
    def next_study2(self):
        study2idx = self.permutation_study2[self.ptr_study2]
        # initialize the graph
        self.study2[study2idx].reset(self.graphs["graph1"])
        self.instance = self.study2[study2idx]
        self.ptr_study2 += 1
    def reset_ptr_study2(self):
        self.permutation_study2 = np.random.permutation(self.numstudy2)
        self.ptr_study2 = 0
    # Sample new problem for study3
    def next_study3(self):
        study3idx = self.permutation_study3[self.ptr_study3]
        # initialize the graph
        self.study3[study3idx].reset(self.graphs["graph1"])
        self.instance = self.study3[study3idx]
        self.ptr_study3 += 1
    def reset_ptr_study3(self):
        self.permutation_study3 = np.random.permutation(self.numstudy3)
        self.ptr_study3 = 0
