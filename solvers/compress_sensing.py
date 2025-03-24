from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from mri.operators import WaveletUD2
    from mri.reconstructors import SingleChannelReconstructor
    from modopt.opt.proximity import SparseThreshold
    from modopt.opt.linear import Identity

# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'CompreS'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "optimizer": ["pogm", "fista", "condat-vu"],
        "wavelet_name": ["haar", "sym8"],
        "nb_scales": [4],
        "lambd": [1e-7, 1e-8, 1e-4],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = ["pip:pysap-python", "pip:modopt"]

    def set_objective(self, kspace, foperator, gt):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.kspace, self.foperator, self.gt = kspace, foperator, gt
        lop = WaveletUD2(wavelet_id=24, nb_scale=4)
        rop = SparseThreshold(Identity(), 2e-8, thresh_type="soft")
        self.reconstructor = SingleChannelReconstructor(
            fourier_op=self.foperator,
            linear_op=lop,
            regularizer_op=rop,
            gradient_formulation='synthesis',
            verbose=1,
        )
        

    def run(self):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html
        image, costs, metrics = self.reconstructor.reconstruct(
            kspace_data=self.kspace,
            optimization_alg='fista',
            num_iterations=200,
        )  
        self.beta = image

        

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(beta=self.beta)
