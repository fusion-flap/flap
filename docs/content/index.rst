.. These will be included in the navigation bar on the left

.. toctree::
   :hidden:
   :maxdepth: 1

   self
   Examples Gallery <auto_examples/index>
   User's Guide <users_guide/index>
   Extensions <extensions>
   Developer's Guide <developers_guide>
   API reference <flap>
   GitHub <http://github.com/fusion-flap/flap>


FLAP - Fusion Library of Analysis Programs
==========================================

This package is intended for analysing large multidimensional datasets.

Install
-------
After downloading FLAP from `GitHub <http://github.com/fusion-flap/flap>`_, set up an Anaconda environment using the appropriate, platform-specific ``.yml`` file provided for either Linux or Windows.

For example, on Windows, from the Anaconda Prompt run:

.. code:: bash

   $ conda env create -f docs/flap_windows.yml

Activate the new ``flap`` environment:

.. code:: bash

   $ conda activate flap

Install the package:

.. code:: bash

   $ python setup.py install

FLAP is now ready to use!


Getting started
---------------

To get started, see the :doc:`Examples Gallery <auto_examples/index>`.

For more information, see the :doc:`User's Guide <users_guide>`.

Extensions
----------
:doc:`Various extensions <extensions>` are available for FLAP.

API reference
-------------

For developers, a :doc:`Guide <developers_guide>` and an :doc:`API reference <flap>` is available.


Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
