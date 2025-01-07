.. These will be included in the navigation bar on the left

.. toctree::
   :hidden:
   :maxdepth: 1

   self
   Examples Gallery <auto_examples/index>
   User's Guide <users_guide/index>
   Extensions <extensions>
   Developer's Guide <developers_guide>
   Presentations <presentations>
   API reference <flap>
   UML diagrams <uml_diagrams>
   GitHub <http://github.com/fusion-flap/flap>


FLAP - Fusion Library of Analysis Programs
==========================================

This package is intended for analysing large multidimensional datasets.

.. _install:

Install
-------
First, clone the FLAP repository from `GitHub <http://github.com/fusion-flap/flap>`_:

.. code:: bash

   $ git clone https://github.com/fusion-flap/flap.git

Navigate to the newly created ``flap`` folder:

.. code:: bash

   $ cd flap

Before installing FLAP, first set up an Anaconda/Mamba environment using the appropriate, platform-specific ``.yml`` file provided for either Linux or Windows.

.. note:: 

   **Conda and Mamba**
   
   Conda is the environment handling tool in `Anaconda  <https://anaconda.org/>`_. From 2024 conda is free only for enterprises below 200 staff. Mamba is a free
   alternative providing the same functionality. 
   
   Conda can be downloaded and installed from `anaconda.org <https://anaconda.org/>`_.
   
   Mamba is contained in Miniforge, which can be downloaded from `conda-forge <https://conda-forge.org/download/>`_. Download the package for your operating system and 
   follow the instructions for installation.
   

For example, on Windows, from the Anaconda Prompt/Miniforge Prompt run(``mamba`` should be substituted for ``conda`` if Mamba is used):

.. code:: bash

   $ conda env create -f docs/flap_windows.yml

Activate the new ``flap`` environment:

.. code:: bash

   $ conda activate flap

Install the package:

.. code:: bash

   $ python -m pip install .

.. tip::

   To install FLAP in editable mode (e.g. for development), use the ``-e`` flag:

   .. code:: bash

      $ python -m pip install -e .

   This installs FLAP in such a way that any changes made in the source code are instantly reflected in any software using it. (This is similar to adding the package directory to your ``$PATH``, but more robust.)

.. tip::
   The default environment uses ``numpy >= 2.0``. In case this is incompatible with some additional library needed, ``*_oldnumpy.yml`` files are also available in the ``docs/`` folder, which can be used to create environments with an older version of numpy.

FLAP is now ready to use!


Getting started
---------------

To get started, see the :doc:`Examples Gallery <auto_examples/index>`.

For more information, see the :doc:`User's Guide <users_guide/index>`.

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
