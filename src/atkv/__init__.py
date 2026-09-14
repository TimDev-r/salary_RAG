"""AT-KV assistant: retrieval over the Austrian IT collective agreement.

WHY THIS FILE SETS AN ENVIRONMENT VARIABLE
------------------------------------------
faiss and torch each bundle their own OpenMP runtime. On macOS, once faiss has
actually USED its runtime (building an index, not merely being imported),
loading a torch model afterwards segfaults the process: exit code 139, no
traceback, no Python-level exception, nothing in the logs. A pipeline that
indexes and then reranks hits this every time.

OMP_NUM_THREADS=1 avoids it. Isolation testing showed:

    import faiss, then a torch model            -> SEGFAULT
    torch model first, then faiss               -> works
    OMP_NUM_THREADS=1                           -> works
    KMP_DUPLICATE_LIB_OK=TRUE alone             -> SEGFAULT

Import order alone is not enough, because the crash is triggered by faiss
USING its runtime, not by the import. It has to be set before either library
initialises, which means here, at package import, ahead of any submodule.

setdefault, not assignment: an operator who knows their environment can
override it. The corpus is ~500 chunks, so single-threaded BLAS costs nothing
measurable here; on a much larger index this would deserve revisiting.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
