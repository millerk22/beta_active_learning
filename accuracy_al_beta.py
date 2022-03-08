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
        for num, acc_model_name in enumerate(MODELS.keys()):
            modelname_dir = os.path.join(RESULTS_DIR, acc_model_name)
            if not os.path.exists(modelname_dir):
                os.makedirs(modelname_dir)

            choices_fnames = glob(os.path.join(RESULTS_DIR, "choices_*.npy"))
            def compute_accuracies(choices_fname, show_tqdm):
                # get copy of model on this cpu
                model = deepcopy(MODELS[acc_model_name])
                acq_func_name, modelname = choices_fname.split("_")[-2:]
                modelname = modelname.split(".")[0]

                choices = np.load(choices_fname)
                labeled_ind = np.load(os.path.join(RESULTS_DIR, "init_labeled.npy"))

                if show_tqdm:
                    iterator_object = tqdm(range(labeled_ind.size,choices.size), desc=f"Computing Acc of {acq_func_name}-{modelname}")
                else:
                    iterator_object = range(labeled_ind.size,choices.size)

                acc = np.array([])
                for j in iterator_object:
                    train_ind = choices[:j]
                    u = model.fit(train_ind, labels[train_ind])
                    acc = np.append(acc, gl.ssl.ssl_accuracy(model.predict(), labels, train_ind.size))

                acc_dir = os.path.join(RESULTS_DIR, acc_model_name)
                if not os.path.exists(acc_dir):
                    os.makedirs(acc_dir)

                np.save(os.path.join(acc_dir, f"acc_{acq_func_name}_{modelname}.npy"), acc)
                return

            print(f"-------- Computing Accuracies in {acc_model_name}, {num+1}/{len(MODELS)} in {RESULTS_DIR} ({out_num+1}/{len(results_directories)}) -------")
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
    acc_model_names_list = np.unique([fpath.split("/")[-1] for fpath in results_models_directories])
    for acc_model_name in tqdm(acc_model_names_list, desc=f"Saving results over all runs to: {overall_results_dir}", total=len(acc_model_names_list)):
        overall_results_file = os.path.join(overall_results_dir, f"{acc_model_name}_stats.csv")
        print(os.path.join("results", f"{args.dataset}_results_*_{args.iters}", f"{acc_model_name}", "accs.csv"))
        acc_files = glob(os.path.join("results", f"{args.dataset}_results_*_{args.iters}", f"{acc_model_name}", "accs.csv"))
        dfs = [pd.read_csv(f) for f in sorted(acc_files)]
        print(dfs)
        possible_columns = reduce(np.union1d, [df.columns for df in dfs])
        all_columns = {}
        for col in possible_columns:
            vals = np.array([df[col].values for df in dfs if col in df.columns])
            all_columns[col + " : avg"] = np.average(vals, axis=0)
            all_columns[col + " : std"] = np.std(vals, axis=0)

        all_df = pd.DataFrame(all_columns)
        all_df.to_csv(overall_results_file, index=None)
    print("-"*40)
