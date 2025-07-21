======
HiPoSa
======

.. image:: https://img.shields.io/pypi/v/hiposa.svg
        :target: https://pypi.org/project/hiposa

.. image:: https://img.shields.io/travis/phzwart/hiposa.svg
        :target: https://travis-ci.org/phzwart/hiposa

.. image:: https://readthedocs.org/projects/hiposa/badge/?version=latest
        :target: https://hiposa.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Hierarchical Poisson Sampling
============================

HiPoSa is a Python library for generating hierarchical Poisson disk sampling patterns with support for symmetry operations, tiling, and multi-dimensional spaces.

Features
--------

* **Multi-dimensional Support**: Works in 2D, 3D, and up but is optimized for 2D.
* **Hierarchical Sampling**: Generate samples at multiple spacing levels
* **Symmetry Operations**: Support for rotational, translational, and custom symmetries
* **Periodic Tiling**: Create seamless tiling patterns for large areas
* **Point Selection**: Intelligent point selection based on interpolated data
* **High Performance**: Optimized algorithms with KDTree for efficient neighbor searches

Installation
------------

.. code-block:: bash

    pip install hiposa

For development installation:

.. code-block:: bash

    git clone https://github.com/phzwart/hiposa.git
    cd hiposa
    pip install -e ".[dev]"

Quick Start
-----------

Basic Poisson Disk Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from hiposa import PoissonDiskSamplerWithExisting

    # Define the domain
    domain = [(0, 10), (0, 10)]  # 10x10 square
    
    # Minimum distance between points
    r = 0.5
    
    # Create sampler and generate points
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
    points, labels = sampler.sample()
    
    print(f"Generated {len(points)} points")

Hierarchical Tiling
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from hiposa import PoissonTiler
    
    # Define a tile size and a series of spacing levels
    tile_size = 10.0
    spacings = [2.0, 1.0, 0.5]  # From largest to smallest
    
    # Create a tiler
    tiler = PoissonTiler(tile_size=tile_size, spacings=spacings)
    
    # Get points in a larger region
    region = ((0, 50), (0, 30))  # 50x30 rectangle
    points, levels = tiler.get_points_in_region(region)
    
    print(f"Generated {len(points)} points across {len(spacings)} levels")

Symmetry Operations
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from hiposa import PoissonDiskSamplerWithExisting

    # Define a rotational symmetry operator
    def rotate_90_degrees(point):
        x, y = point
        return np.array([-y, x])

    # Create sampler with symmetry
    domain = [(0, 10), (0, 10)]
    sampler = PoissonDiskSamplerWithExisting(
        domain=domain, 
        r=0.5,
        symmetry_operators=[rotate_90_degrees]
    )
    
    points, labels = sampler.sample()
    print(f"Generated {len(points)} points with symmetry")

Point Selection
~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from hiposa import PointSelector

    # Define a function to evaluate
    def f_function(point):
        x, y = point
        return np.sin(x) * np.cos(y)

    # Create point selector
    xy = np.random.rand(100, 2) * 10
    levels = np.random.randint(0, 3, 100)
    scales = [1.0, 0.5, 0.25]
    
    selector = PointSelector(
        xy=xy,
        levels=levels,
        scales=scales,
        f_function=f_function,
        grid_x=np.linspace(0, 10, 50),
        grid_y=np.linspace(0, 10, 50)
    )
    
    # Run selection
    selected_points = selector.run(max_level=2)
    print(f"Selected {len(selected_points)} points")

Documentation
------------

Full documentation is available at https://hiposa.readthedocs.io

API Reference
~~~~~~~~~~~~~

* :class:`PoissonDiskSamplerWithExisting`: Core Poisson disk sampling class
* :class:`PoissonTiler`: Hierarchical tiling with multiple spacing levels
* :class:`PointSelector`: Intelligent point selection based on interpolated data

Examples
--------

See the `examples/` directory for Jupyter notebooks demonstrating:

* Basic sampling and tiling
* Symmetry operations
* Point selection algorithms
* Performance benchmarks
* Visualization techniques

Contributing
------------

We welcome contributions! Please see our `Contributing Guide <CONTRIBUTING.rst>`_ for details.

Development
~~~~~~~~~~~

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: ``pytest``
6. Submit a pull request

Testing
~~~~~~~

.. code-block:: bash

    # Run all tests
    pytest

    # Run with coverage
    pytest --cov=hiposa

    # Run specific test file
    pytest tests/test_basic_sampling.py

License
-------

This project is licensed under the MIT License - see the `LICENSE` file for details.

Credits
-------

This package was created with Cookiecutter_ and the `audreyfeldroy/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyfeldroy/cookiecutter-pypackage`: https://github.com/audreyfeldroy/cookiecutter-pypackage

