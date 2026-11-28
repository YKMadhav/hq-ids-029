# Direction 3: QUBO sub-problems of an IDPS - pre-registration (written before any run)

Task: feature-subset selection as QUBO (mRMR form) on the de-duplicated UNSW-NB15 5k sample.
  E(x) = - sum_i rel_i x_i + ALPHA * sum_{i<j} red_ij x_i x_j + LAM * (sum x - k)^2,  x in {0,1}^n
  rel_i = normalised mutual information with label (train split), red_ij = |Spearman| between features,
  ALPHA=1, LAM=0.5, k=n//3. Instances: random feature subsets of size n in {10,14,18}, 8 instances/size (seeded).
Methods: exact brute force (ground truth), greedy forward, simulated annealing (SA), tabu search, random sampling,
  QAOA p=1,2,3 (exact statevector simulation, 5 random restarts, COBYLA, 1000 shots).
Metrics per instance: P(optimum) in the final sample, approximation ratio of best-found energy (0=bad, 1=optimal),
  wall-clock, number of energy evaluations. Downstream: XGBoost accuracy (5 seeds) on the selected subset.
Fairness rule: classical solvers get AT LEAST the wall-clock of QAOA's total optimisation+sampling time per instance.
  QAOA times are simulator times (not hardware); a real device would not be faster for statevector-level problems this small.
Success criterion (fixed now): the quantum layer "helps" only if QAOA beats SA/tabu at equal wall-clock on
  P(optimum) or best-energy gap in a majority of instances at n=18 with paired bootstrap CI excluding 0, AND the selected
  subset gives >= classical accuracy. Expected outcome [Likely]: classical reaches the optimum on every instance
  (n<=18 is tiny), so the criterion fails; the informative output is scaling of the QAOA/classical gap with n.
Limits: n<=18 is the simulator ceiling here; feature selection is a model, not a detection result; heuristics' hyperparameters
  are fixed in code before the run.
