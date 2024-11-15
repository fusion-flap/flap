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

Before installing FLAP, first set up an Anaconda environment using the appropriate, platform-specific ``.yml`` file provided for either Linux or Windows.

For example, on Windows, from the Anaconda Prompt run:

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
