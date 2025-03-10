#
# Copyright (c) 2024, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cv2 as cv
import cupy as cp
import numpy as np

from typing import List

from .plugin_manager import PluginBase


class Extremize(PluginBase):
    """
    Args:
        kernel_size (int): Size of the erosion and dilation kernel. Default is 3, which means a 3x3 square kernel.
        iterations (int): Number of times erosion/dilation is applied. Default is 1.
        **kwargs (): Additional keyword arguments.
    """

    def __init__(
        self,
        input_layer_name="inpaint",
        kernel_size: int = 3,
        iterations: int = 1,
        reverse: bool = False,
        default_layer_name: str = "inpaint",
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
        # Convert the elevation map to a format suitable for erosion (if necessary)
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
                print(f"No layers are found, using inpaint!")
                layer_data = self.get_layer_data(
                    elevation_map,
                    layer_names,
                    plugin_layers,
                    plugin_layer_names,
                    semantic_map,
                    semantic_layer_names,
                    "inpaint",
                )
        layer_np = cp.asnumpy(layer_data)

        mu = np.mean(layer_np)
        sigma = np.std(layer_np) + 0.02
        # Define thresholds for erosion and dilation
        lower_threshold = mu - 0.5 * sigma  # Erode values below this
        upper_threshold = mu + 0.5 * sigma  # Dilate values above this
        mask_erosion = layer_np < lower_threshold
        mask_dilation = layer_np > upper_threshold
        # Define the erosion kernel
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        if self.reverse:
            layer_np = 1 - layer_np
        # Apply erosion and dilation 
        layer_min = float(layer_np.min())
        layer_max = float(layer_np.max())
        layer_np_normalized = ((layer_np - layer_min) * 255 / (layer_max - layer_min)).astype("uint8")

        eroded_norm = cv.erode(layer_np_normalized, kernel, iterations=self.iterations)
        dilated_norm = cv.dilate(layer_np_normalized, kernel, iterations=self.iterations)
        output_norm = layer_np_normalized.copy()
        output_norm[mask_erosion] = eroded_norm[mask_erosion]
        output_norm[mask_dilation] = dilated_norm[mask_dilation]
        final_map = output_norm.astype(np.float32) * (layer_max - layer_min) / 255 + layer_min
        if self.reverse:
            final_map = 1 - final_map

        # Convert back to cupy array and return
        return cp.asarray(final_map)