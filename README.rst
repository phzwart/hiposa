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


* Free software: MIT license
* Documentation: https://hiposa.readthedocs.io.

Features
--------

Construct hierarchical Poisson Sampling sets that cover a N-D space uniformly.


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

Credits
-------

This package was created with Cookiecutter_ and the `audreyfeldroy/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyfeldroy/cookiecutter-pypackage`: https://github.com/audreyfeldroy/cookiecutter-pypackage

