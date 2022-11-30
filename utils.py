import graphlearning as gl
import os
import numpy as np
from copy import deepcopy


def get_models(G, model_names):
    MODELS = {'poisson':gl.ssl.poisson(G),  # poisson learning
              'laplace':gl.ssl.laplace(G), # laplace learning
              'beta1000': beta_learning(G, tau=0.1),
              'beta0100': beta_learning(G, tau=0.01),
              'beta0010': beta_learning(G, tau=0.001),
              'beta0001': beta_learning(G, tau=0.0001),
              'beta1000eps1000': beta_learning(G, tau=0.1, eps=0.1),
              'beta0100eps1000': beta_learning(G, tau=0.01, eps=0.1),
              'beta0010eps1000': beta_learning(G, tau=0.001, eps=0.1),
              'beta0001eps1000': beta_learning(G, tau=0.0001, eps=0.1),
              'beta1000eps0100': beta_learning(G, tau=0.1, eps=0.01),
              'beta0100eps0100': beta_learning(G, tau=0.01, eps=0.01),
              'beta0010eps0100': beta_learning(G, tau=0.001, eps=0.01),
              'beta0001eps0100': beta_learning(G, tau=0.0001, eps=0.01)
              }

    return [deepcopy(MODELS[name]) for name in model_names]


def load_graph(dataset, metric, numeigs=200, data_dir="data", returnX=False, returnK=False):
    X, clusters = gl.datasets.load(dataset.split("-")[0], metric=metric)
    if dataset.split("-")[-1] == 'evenodd':
        labels = clusters % 2
    elif dataset.split("-")[-1][:3] == "mod":
        modnum = int(dataset[-1])
        labels = clusters % modnum
    else:
        labels = clusters

    if dataset.split("-")[0] == 'mstar': # allows for specific train/test set split TODO
        trainset = None
#     elif metric == "hsi":
#         trainset = np.where(labels != 0)[0]
    else:
        trainset = None

    # Construct the similarity graph
    print(f"Constructing similarity graph for {dataset}")
    knn = 20
    if dataset == 'isolet':
        print("Using knn = 5 for Isolet")
        knn = 5
    graph_filename = os.path.join(data_dir, f"{dataset.split('-')[0]}_{knn}")

    normalization = "combinatorial"
    method = "lowrank"
    if dataset.split("-")[0] in ["mnist", "fashionmnist", "cifar", "emnist", "mnistsmall", "fashionmnistsmall", "salinassub", "paviasub"]:
        normalization = "normalized"
    if labels.size < 100000:
        method = "exact"
    
    print(f"Eigendata calculation will be {method}")

    try:
        G = gl.graph.load(graph_filename)
        found = True
    except:
        if metric == "hsi":
            W = gl.weightmatrix.knn(X, knn, similarity="angular") # LAND does 100 in HSI
        else:
            W = gl.weightmatrix.knn(X, knn)
        G = gl.graph(W)
        found = False

    if numeigs is not None:
        eigdata = G.eigendata[normalization]['eigenvalues']
        if eigdata is not None:
            prev_numeigs = eigdata.size
            if prev_numeigs >= numeigs:
                print("Retrieving Eigendata...")
            else:
                print(f"Requested {numeigs} eigenvalues, but have only {prev_numeigs} stored. Recomputing Eigendata...")
        else:
            print(f"No Eigendata found, so computing {numeigs} eigenvectors...")

        evals, evecs = G.eigen_decomp(normalization=normalization, k=numeigs, method=method)


    G.save(graph_filename)
    
    if returnX:
        return G, labels, trainset, normalization, X
    
    if returnK:
        return G, labels, trainset, normalization, np.unique(clusters).size
    
    return G, labels, trainset, normalization


def get_active_learner_eig(G, labeled_ind, labels, acq_func_name, gamma=0.1, normalization='combinatorial'):
    numeigs = 100 # default
    if len(acq_func_name.split("-")) > 1:
        numeigs = int(acq_func_name.split("-")[-1])

    # determine if need to recompute eigenvalues/vectors -- Need to still debug what was going wrong with cached evecs and evals
    recompute = True
    if G.eigendata[normalization]['eigenvalues'] is not None:
        if G.eigendata[normalization]['eigenvalues'].size < numeigs:
            recompute = True
    else:
        recompute = True

    if not recompute:
        # Current gl.active_learning is implemented only to allow for exact eigendata compute for "normalized"
        print(f"Using previously stored {normalization} eigendata with {numeigs} evals for computing {acq_func_name}")
        active_learner = gl.active_learning.active_learning(G, labeled_ind.copy(), labels[labeled_ind], eval_cutoff=None)
        
        active_learner.evals = G.eigendata[normalization]['eigenvalues'][:numeigs]
        active_learner.evecs = G.eigendata[normalization]['eigenvectors'][:,:numeigs] 
        if acq_func_name.split("-")[0][-1] == "1":
            active_learner.evals = active_learner.evals[1:numeigs] 
            active_learner.evecs = active_learner.evecs[:,1:numeigs]
        
        active_learner.gamma = gamma
        active_learner.cov_matrix = np.linalg.inv(np.diag(active_learner.evals) + active_learner.evecs[active_learner.current_labeled_set,:].T @ active_learner.evecs[active_learner.current_labeled_set,:] / active_learner.gamma**2.)
        active_learner.init_cov_matrix = active_learner.cov_matrix
    else:
        print("Warning: Computing eigendata with gl.active_learning defaults...")
        active_learner = gl.active_learning.active_learning(G, labeled_ind.copy(), labels[labeled_ind], \
                    eval_cutoff=numeigs, gamma=gamma)
        
        
        if acq_func_name.split("-")[0][-1] == "1":
            active_learner.evals = active_learner.evals[1:] 
            active_learner.evecs = active_learner.evecs[:,1:]
            active_learner.cov_matrix = np.linalg.inv(np.diag(active_learner.evals) +  active_learner.evecs[active_learner.current_labeled_set,:].T @ active_learner.evecs[active_learner.current_labeled_set,:] / active_learner.gamma**2.)
            active_learner.init_cov_matrix = active_learner.cov_matrix
        
    print(f"{acq_func_name}, {active_learner.evals.size}")
    return active_learner





class beta_learning(gl.ssl.ssl):
    def __init__(self, W=None, class_priors=None, tau=0.0, epsK=None):
        """Beta Learning with Epsilon prior
        ===================

        Semi-supervised learning
        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        propagation : str, default='poisson'
            Which propagation from labeled points to use for alpha and beta. Possibilities include: 'poisson' and 'spectral'
        K : int, default=10
            Number of "known" clusters in the dataset. Parameter for choosing epsilon prior size
        """
        super().__init__(W, class_priors)
        self.tau = tau
        self.train_ind = np.array([])
        
        
        if epsK is not None:
            # epsilon calculation
            MULT = 5
            num_eps_props = epsK * MULT
            rand_inds = np.random.choice(self.graph.num_nodes, num_eps_props, replace=False)
            props = self.poisson_prop(rand_inds)
            props_to_inds = np.max(props, axis=1)
            epsilons = np.array([np.percentile(props[:,i], 100.*(epsK-1.)/epsK) for i in range(props.shape[1])])
            self.eps = np.max(epsilons)
        else:
            # prior is Beta(1,1,...,1)
            self.eps = 1.0
        
        #Setup accuracy filename
        fname = '_beta' 
        if self.tau > 0:
            fname += '_' + str(int(self.tau * 1e4)).zfill(4)
        self.name = f'Beta Learning Eps = {self.eps}, tau = {self.tau}'

        self.accuracy_filename = fname
        

    def _fit(self, train_ind, train_labels, all_labels=None):
        # Not currently designed for repeated indices in train_ind
        if train_ind.size >= self.train_ind.size:
            mask = ~np.isin(train_ind, self.train_ind)
            prop_ind = train_ind[np.where(mask)[0]]
            prop_labels = train_labels[np.where(mask)[0]]
        else: # if give fewer training labels than before, we assume that this is a "new" instantiation
            prop_ind, prop_labels = train_ind, train_labels
            mask = np.ones(3, dtype=bool)
        self.train_ind = train_ind
        n, nc = self.graph.num_nodes, np.unique(train_labels).size
        
        P = self.poisson_prop(prop_ind)
        
        P /= P[prop_ind,np.arange(prop_ind.size)][np.newaxis,:] # scale by the value at the point sources

        if mask.all(): # prop_ind == train_ind, so all inds are "new"
            self.A = self.eps*np.ones((n, nc))  # Dir(1,1,1,...,1) prior on each node

        # Add propagations according to class for the propagation inds (prop_inds)
        for c in np.unique(prop_labels):
            self.A[:, c] += np.sum(P[:,np.where(prop_labels == c)[0]], axis=1) # sum propagations together according to class
        

        u = self.A / (self.A.sum(axis=1)[:,np.newaxis]) # mean estimator
        return u
    
    def poisson_prop(self, inds):
        # Poisson propagation
        n, num_prop = self.graph.num_nodes, inds.size
        F = np.zeros((n, num_prop))
        F[inds,:] = np.eye(num_prop)
        F -= np.mean(F, axis=0)

        L = self.graph.laplacian()
        if self.tau  > 0.0:
            L += self.tau*sparse.eye(L.shape[0])

        prop = gl.utils.conjgrad(L, F, tol=1e-9)
        prop -= np.min(prop, axis=0)
        return prop



