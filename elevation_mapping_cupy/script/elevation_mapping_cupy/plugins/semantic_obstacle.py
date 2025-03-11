#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
from typing import List
import re

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class SemanticObstacle(PluginBase):
    """This is a filter to create the confidence map of the max class probabilities.

    Args:
        cell_n (int): width and height of the elevation map.
        classes (list): List of classes for semantic filtering. Default is ["person", "grass"].
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self, cell_n: int = 100, classes: list = ["person", "grass"], **kwargs,
    ):
        super().__init__()
        self.indices = []
        self.classes = classes

    def get_layer_indices(self, layer_names: List[str]) -> List[int]:
        """ Get the indices of the layers that are to be processed using regular expressions.
        Args:
            layer_names (List[str]): List of layer names.
        Returns:
            List[int]: List of layer indices.
        """
        indices = []
        for i, layer_name in enumerate(layer_names):
            if any(re.match(pattern, layer_name) for pattern in self.classes):
                indices.append(i)
        return indices

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        rotation,
        elements_to_shift,
        *args,
    ) -> cp.ndarray:
        """

        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            semantic_map (elevation_mapping_cupy.semantic_map.SemanticMap):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """
        # get indices of all layers that contain semantic class information
        data = []
        for m, layer_names in zip(
            [elevation_map, plugin_layers, semantic_map], [layer_names, plugin_layer_names, semantic_layer_names]
        ):
            layer_indices = self.get_layer_indices(layer_names)
            if len(layer_indices) > 0:
                data.append(m[layer_indices])
        if len(data) > 0:
            data = cp.concatenate(data, axis=0)
            confmap = cp.max(data, axis=0)
        else:
            confmap = cp.zeros_like(elevation_map[0])
        return confmap
