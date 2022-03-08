import numpy as np
import matplotlib.pyplot as plt
import graphlearning as gl
import scipy.sparse as sparse
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
import os
from copy import deepcopy
from glob import glob
from scipy.special import softmax
from functools import reduce
from gl_models import beta_learning
from acquisitions import *


from joblib import Parallel, delayed

ACQS = {'unc': unc,
        'uncsftmax':uncsftmax,
        'uncdist':uncdist,
        'uncsftmaxnorm':uncsftmaxnorm,
        'uncnorm':uncnorm,
        'vopt':vopt,
        'mc':mc,
        'mcvopt':mcvopt,
        'random':random,
        'betavar':beta_var}







if __name__ == "__main__":
    parser = ArgumentParser(description="Compute Accuracies in Parallel of Active Learning Tests for Beta Learning")
    parser.add_argument("--dataset", type=str, default='mnist-evenodd')
    parser.add_argument("--metric", type=str, default='vae')
    parser.add_argument("--numcores", type=int, default=6)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()


    X, labels = gl.datasets.load(args.dataset.split("-")[0], metric=args.metric)
    if args.dataset.split("-")[-1] == 'evenodd':
        labels = labels % 2

    nc = np.unique(labels).size

    if args.dataset.split("-")[0] == 'mstar': # allows for specific train/test set split TODO
        trainset = None
    else:
        trainset = None

    # Construct the similarity graph
    print(f"Constructing similarity graph for {args.dataset}")
    knn = 20
    graph_filename = os.path.join("data", f"{args.dataset.split('-')[0]}_{knn}")
    try:
        G = gl.graph.load(graph_filename)
    except:
        W = gl.weightmatrix.knn(X, knn)
        G = gl.graph(W)
        if np.isin(acq_funcs_names, ["mc", "vopt", "mcvopt"]).any():
            print("Computing Eigendata...")
            evals, evecs = G.eigen_decomp(normalization="combinatorial", k=args.numeigs, method="lowrank", q=150, c=50)
        G.save(graph_filename)

    # MODELS = {'poisson':gl.ssl.poisson(G),  # poisson learning
    #             'beta0':beta_learning(G),  # reweighted laplace learning, tau = 0
    #             'beta01':beta_learning(G, tau=0.01),  # reweighted laplace learning, tau = 0.01
    #              'beta1':beta_learning(G, tau=0.1)}   # reweighted laplace learning, tau = 0.1
    MODELS = {'poisson':gl.ssl.poisson(G),  # poisson learning
                'beta0':beta_learning(G)}


    results_directories = glob(os.path.join("results", f"{args.dataset}_results_*_{args.iters}/"))

    for out_num, RESULTS_DIR in enumerate(results_directories):
        for num, model_name in enumerate(MODELS.keys()):
            modelname_dir = os.path.join(RESULTS_DIR, model_name)
            if not os.path.exists(modelname_dir):
                os.makedirs(modelname_dir)

            choices_fnames = glob(os.path.join(RESULTS_DIR, "choices_*.npy"))
            def compute_accuracies(choices_fname, show_tqdm):
                # get copy of model on this cpu
                model = deepcopy(MODELS[model_name])

                # check if test already completed previously
                acc_run_savename = os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy")
                if os.path.exists(acc_run_savename):
                    print(f"Found choices for {acq_func_name} in {model_name}")
                    return

                choices = np.load(choices_fname)
                train_ind = np.load(os.path.join(RESULTS_DIR, "init_labeled.npy"))
                u = model.fit(train_ind, labels[train_ind])
                acc = np.array([gl.ssl.ssl_accuracy(model.predict(), labels, train_ind.size)])

                if show_tqdm:
                    iterator_object = tqdm(range(train_ind.size,choices.size), desc=f"{args.dataset} test {it+1}/{args.numtests}, seed = {seed}")
                else:
                    iterator_object = range(train_ind.size,choices.size)

                for j in iterator_object:
                    if trainset is None:
                        candidate_set = np.delete(np.arange(G.num_nodes), train_ind)
                    else:
                        candidate_set = np.delete(trainset, train_ind)

                    if acq_func_name == "random":
                        k = np.random.choice(candidate_set)
                    else:
                        if acq_func_name in ["betavar"]:
                            acq_func_vals = acq_func(model.A, candidate_set)
                        elif acq_func_name in ["mc", "mcvopt", "vopt"]:
                            C_a = np.linalg.inv(np.diag(evals) + evecs[train_ind,:].T @ evecs[train_ind,:] / args.gamma**2.)
                            acq_func_vals = acq_func(u, C_a, evecs, gamma=args.gamma)
                        else:
                            acq_func_vals = acq_func(u)

                        # active learning query choice
                        acq_func_vals = acq_func_vals[candidate_set]
                        maximizer_inds = np.where(np.isclose(acq_func_vals, acq_func_vals.max()))[0]
                        k = candidate_set[np.random.choice(maximizer_inds)]


                    # oracle and model update
                    train_ind = np.append(train_ind, k)
                    u = model.fit(train_ind, labels[train_ind])
                    acc = np.append(acc, gl.ssl.ssl_accuracy(model.predict(), labels, train_ind.size))

                acc_dir = os.path.join(RESULTS_DIR, model_name)
                if not os.path.exists(acc_dir):
                    os.makedirs(acc_dir)
                np.save(os.path.join(acc_dir, f"acc_{acq_func_name}_{model_name}.npy"), acc)
                np.save(os.path.join(RESULTS_DIR, f"choices_{acq_func_name}_{model_name}.npy"), train_ind)
                return

            print(f"--Accuracies for {model_name}, {num+1}/{len(MODELS)} in {RESULTS_DIR} ({out_num+1}/{len(results_directories)})--")
            # show_bools is for tqdm iterator to track progress of some
            show_bools = np.zeros(len(choices_fnames), dtype=bool)
            show_bools[::args.numcores] = True

            Parallel(n_jobs=args.numcores)(delayed(compute_accuracies)(choices_fname, show) for choices_fname, show \
                    in zip(choices_fnames, show_bools))
            print()

        # Consolidate results
        print(f"Consolidating accurary results of run in: {os.path.join(RESULTS_DIR)}...")
        for modelname_dir in glob(os.path.join(RESULTS_DIR, "*/")):
            accs_fnames = glob(os.path.join(modelname_dir, "acc_*.csv"))

            columns = {}
            for fname in accs_fnames:
                acc = np.load(fname)
                acq_func_name, modelname = fname.split("_")[-2:]
                modelname = modelname.split(".")[0]
                columns[acq_func_name + " : " + modelname] = acc
            acc_df = pd.DataFrame(columns)
            acc_df.to_csv(os.path.join(modelname_dir, "accs.csv"), index=None)

        print("-"*40)
        print("-"*40)
    print()


    # Get average and std curves over all tests
    overall_results_dir = os.path.join("results", f"{args.dataset}_overall_{args.iters}")
    if not os.path.exists(overall_results_dir):
        os.makedirs(overall_results_dir)

    results_models_directories = glob(os.path.join("results", f"{args.dataset}_results_*_{args.iters}", "*/"))
    model_names_list = np.unique([fpath.split("/")[-1] for fpath in results_models_directories])
    for model_name in tqdm(model_names_list, desc=f"Saving results over all runs to: {overall_results_dir}", total=len(model_names_list)):
        overall_results_file = os.path.join(overall_results_dir, f"{model_name}_stats.csv")
        acc_files = glob(os.path.join("results", f"{args.dataset}_results_*_{args.iters}", f"{model_name}", "accs.csv"))
        dfs = [pd.read_csv(f) for f in sorted(acc_files)]
        possible_columns = reduce(np.union1d, [df.columns for df in dfs])
        all_columns = {}
        for col in possible_columns:
            vals = np.array([df[col].values for df in dfs if col in df.columns])
            all_columns[col + " : avg"] = np.average(vals, axis=0)
            all_columns[col + " : std"] = np.std(vals, axis=0)

        all_df = pd.DataFrame(all_columns)
        all_df.to_csv(overall_results_file, index=None)
    print("-"*40)
