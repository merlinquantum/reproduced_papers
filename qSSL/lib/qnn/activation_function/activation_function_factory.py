# MIT License
#
# Copyright (c) 2021 Ben Jaderberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .null_activation_function import NullActivationFunction
from .partial_meas_activation_function import PartialMeasActivationFunction


class ActivationFunctionFactory:
    def __init__(self, activation_function_type):
        self.activation_function_type = activation_function_type

    def get(self):
        """
        Returns appropriate activation function object.
        """

        if (
            self.activation_function_type == "null"
            or self.activation_function_type is None
        ):
            return NullActivationFunction()
        elif self.activation_function_type == "partial_measurement_half":
            return PartialMeasActivationFunction("half")

        elif (
            self.activation_function_type[: len("partial_measurement_")]
            == "partial_measurement_"
        ):
            n_measurements = int(
                self.activation_function_type[len("partial_measurement_") :]
            )
            return PartialMeasActivationFunction(n_measurements)

        else:
            raise ValueError("Invalid activation function type.")
