from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import h5py as hp
    import pandas as pd
    import matplotlib.pyplot as plt
    import mrinufft


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Knee_train"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {

    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = ["pip:h5py", "pip:mri-nufft"]

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        
        fichier = hp.File("/home/albqn/Documents/CPGE/TIPE/benchmark_inverse_problem/datasets/file1000311.h5", 'r')
        rss = fichier['reconstruction_rss'][20]
        image = rss # ground truth image
        samples_loc = mrinufft.initialize_2D_radial(Nc=640, Ns=544) # voir le header
        density = mrinufft.density.voronoi(samples_loc) # améliore la qualité de l'image, savoir expliquer
        NufftOperator = mrinufft.get_operator("finufft") # choix du backend
        nufft = NufftOperator(samples_loc, shape=(320, 320), n_coils=1,
        density=density) # idem voir le header
        kspace_non_cartesien = nufft.op(rss)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(kspace=kspace_non_cartesien, foperator=nufft, gt=image)
