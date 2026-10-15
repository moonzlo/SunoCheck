import os

# Must run before numpy and scipy load OpenBLAS: each copy otherwise reserves buffers for every CPU thread
# (about 600 MB of commit per copy on 20 threads), and the detector does no heavy linear algebra.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
