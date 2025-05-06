Selective Subdivision AAA
===================================
Divide and conquer strategy for the AAA algorithm. Similar to the strategy proposed in https://doi.org/10.48550/arXiv.2405.19582 for eigenvalue problems.
It divides the region of interest into smaller subregions making sure that poles that are found in the parent region are also found in the subregion.
If the poles have moved significantly the region is subdivided further. This way it is ensured that (almost) all poles are found accurately.

.. autofunction:: diffaaable.selective.selective_subdivision_aaa

.. automodule:: diffaaable.selective
   :members:
