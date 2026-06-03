#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Thin shim for legacy tooling.

All project metadata and dependencies are declared in ``pyproject.toml``,
which is the single source of truth. This file exists only so that tools that
still invoke ``setup.py`` directly continue to work.
"""

from setuptools import setup

setup()
