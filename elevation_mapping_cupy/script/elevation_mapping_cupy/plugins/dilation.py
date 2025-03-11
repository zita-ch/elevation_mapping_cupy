#
# Copyright (c) 2024, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cv2 as cv
import cupy as cp
import numpy as np

from typing import List

from .plugin_manager import PluginBase


class Dilation(PluginBase):
    """
    This class is used for applying dilation to an elevation map or specific layers within it.

    Args:
        kernel_size (int): Size of the dilation kernel. Default is 3, which means a 3x3 square kernel.
        iterations (int): Number of times dilation is applied. Default is 1.
        **kwargs (): Additional keyword arguments.
    """

    def __init__(
        self,
        input_layer_name="max_prob",
        kernel_size: int = 3,
        iterations: int = 1,
        reverse: bool = False,
        default_layer_name: str = "max_prob",
        **kwargs,
    ):
        super().__init__()
        self.input_layer_name = input_layer_name
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.reverse = reverse
        self.default_layer_name = default_layer_name

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        """
        Applies dilation to the given elevation map.

        Args:
            elevation_map (cupy._core.core.ndarray): The elevation map to be dilated.
            layer_names (List[str]): Names of the layers in the elevation map.
            plugin_layers (cupy._core.core.ndarray): Layers provided by other plugins.
            plugin_layer_names (List[str]): Names of the layers provided by other plugins.
            *args (): Additional arguments.

        Returns:
            cupy._core.core.ndarray: The dilated elevation map.
        """
        # Convert the elevation map to a format suitable for dilation (if necessary)
        layer_data = self.get_layer_data(
            elevation_map,
            layer_names,
            plugin_layers,
            plugin_layer_names,
            semantic_map,
            semantic_layer_names,
            self.input_layer_name,
        )
        if layer_data is None:
            print(f"No layers are found, using {self.default_layer_name}!")
            layer_data = self.get_layer_data(
                elevation_map,
                layer_names,
                plugin_layers,
                plugin_layer_names,
                semantic_map,
                semantic_layer_names,
                self.default_layer_name,
            )
            if layer_data is None:
                print(f"No layers are found, using max_prob!")
                layer_data = self.get_layer_data(
                    elevation_map,
                    layer_names,
                    plugin_layers,
                    plugin_layer_names,
                    semantic_map,
                    semantic_layer_names,
                    "max_prob",
                )
        layer_np = cp.asnumpy(layer_data)

        # Define the dilation kernel
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        if self.reverse:
            layer_np = 1 - layer_np
        # Apply dilation
        layer_min = float(layer_np.min())
        layer_max = float(layer_np.max())
        layer_np_normalized = ((layer_np - layer_min) * 255 / (layer_max - layer_min)).astype("uint8")
        dilated_map_np = cv.dilate(layer_np_normalized, kernel, iterations=self.iterations)
        dilated_map_np = dilated_map_np.astype(np.float32) * (layer_max - layer_min) / 255 + layer_min
        if self.reverse:
            dilated_map_np = 1 - dilated_map_np

        # Convert back to cupy array and return
        return cp.asarray(dilated_map_np)
