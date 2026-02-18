import re

import numpy as np
import perceval as pcvl
from perceval.components.unitary_components import BS, PS


# a parametrized quantum circuit with variational and encoding layers
class ParametrizedQuantumCircuit:
    def __init__(
        self,
        m=3,
        arch=None,
        use_clements=False,
    ):
        if arch is None:
            arch = ["var", "var", "var", "enc", "var", "var", "var"]
        self.m = m
        self.arch = arch
        if use_clements:
            self.circuit = self.get_circuit_clements_based()
        else:
            self.circuit = self.get_circuit()

        self.var_params, self.enc_params = self.get_params()
        self.var_param_names = [p.name for p in self.var_params]
        self.enc_param_names = [p.name for p in self.enc_params]

    def get_variational_layer(self, layer_idx):
        modes = self.m
        var = pcvl.Circuit(modes, name="var_" + str(layer_idx))
        # add phase shifters
        for m in range(modes):
            var.add(
                m,
                PS(
                    pcvl.P(
                        "phi_" + str(m) + "_" + str(layer_idx + 1),
                        min_v=0,
                        max_v=2 * np.pi,
                        periodic=True,
                    )
                ),
            )

        # add beam splitters
        for m in range(0, modes - 1, 2):
            var.add(
                (m, m + 1),
                BS(
                    pcvl.P(
                        "theta_" + str(m) + "_" + str(layer_idx + 1),
                        min_v=0,
                        max_v=2 * np.pi,
                        periodic=True,
                    ),
                    pcvl.P(
                        "psi_" + str(m) + "_" + str(layer_idx + 1),
                        min_v=0,
                        max_v=2 * np.pi,
                        periodic=True,
                    ),
                ),
            )
        for m in range(1, modes - 1, 2):
            var.add(
                (m, m + 1),
                BS(
                    pcvl.P(
                        "theta_" + str(m) + "_" + str(layer_idx + 1),
                        min_v=0,
                        max_v=2 * np.pi,
                        periodic=True,
                    ),
                    pcvl.P(
                        "psi_" + str(m) + "_" + str(layer_idx + 1),
                        min_v=0,
                        max_v=2 * np.pi,
                        periodic=True,
                    ),
                ),
            )
        return var

    def get_variational_clements_layer(self, layer_idx):
        modes = self.m
        mode_range = tuple([val.item() for val in np.arange(modes, dtype=int)])
        var_clem = pcvl.Circuit(modes, name="var_clem_" + str(layer_idx))

        # add generic interferometer (Clements based)
        var_clem.add(
            mode_range,
            pcvl.Circuit.generic_interferometer(
                modes,
                lambda i: BS(
                    theta=pcvl.P("theta_" + str(i) + "_" + str(layer_idx + 1)),
                    phi_tr=pcvl.P("psi_" + str(i) + "_" + str(layer_idx + 1)),
                ),
            ),
        )

        return var_clem

    def get_encoding_layer(self, layer_idx, mode_range):
        modes = self.m
        enc = pcvl.Circuit(modes, name="enc_" + str(layer_idx))

        # add phase shifters
        for m in mode_range:
            pcvl.P(
                "enc_" + str(m) + "_" + str(layer_idx + 1),
                min_v=0,
                max_v=2 * np.pi,
                periodic=True,
            )
            enc.add(
                m,
                PS(
                    pcvl.P(
                        "enc_" + str(m) + "_" + str(layer_idx + 1),
                        min_v=0,
                        max_v=2 * np.pi,
                        periodic=True,
                    )
                ),
            )
        return enc

    def get_circuit(self):
        modes = self.m
        mode_range = tuple([val.item() for val in np.arange(modes, dtype=int)])
        active_modes = mode_range
        arch = self.arch
        c = pcvl.Circuit(modes)

        var_layer_num = 0
        enc_layer_num = 0
        for layer_name in arch:
            split = re.split(r"\[|\]", layer_name)
            if len(split) == 1:
                layer_type = split[0]
            else:
                layer_type, modes_type = split[:-1]
                if ":" in modes_type:
                    start, end = modes_type.split(":")
                    active_modes = np.arange(int(start), int(end))
                else:
                    active_modes = np.array(modes_type.split(","), dtype=int)
                active_modes = tuple([val.item() for val in active_modes])

            if layer_type == "var":
                c.add(mode_range, self.get_variational_layer(var_layer_num))
                var_layer_num += 1
            else:
                c.add(mode_range, self.get_encoding_layer(enc_layer_num, active_modes))
                enc_layer_num += 1

        return c

    def get_circuit_clements_based(self):
        modes = self.m
        mode_range = tuple([val.item() for val in np.arange(modes, dtype=int)])
        active_modes = mode_range
        arch = self.arch
        c = pcvl.Circuit(modes)

        var_layer_num = 0
        enc_layer_num = 0
        for layer_name in arch:
            split = re.split(r"\[|\]", layer_name)
            if len(split) == 1:
                layer_type = split[0]
            else:
                layer_type, modes_type = split[:-1]
                if ":" in modes_type:
                    start, end = modes_type.split(":")
                    active_modes = np.arange(int(start), int(end))
                else:
                    active_modes = np.array(modes_type.split(","), dtype=int)
                active_modes = tuple([val.item() for val in active_modes])

            if layer_type == "var":
                c.add(mode_range, self.get_variational_clements_layer(var_layer_num))
                var_layer_num += 1
            else:
                c.add(mode_range, self.get_encoding_layer(enc_layer_num, active_modes))
                enc_layer_num += 1

        return c

    def get_params(self):
        params = self.circuit.get_parameters()
        var_params = []
        enc_params = []
        for p in params:
            if "enc" in p.name:
                enc_params.append(p)
            else:
                var_params.append(p)
        return var_params, enc_params

    def init_params(self, red_factor=1, init_var_params=None):
        if init_var_params is None:
            var_param_map = self.update_var_params(
                np.random.normal(0, 2 * red_factor * np.pi, len(self.var_param_names))
            )
        else:
            var_param_map = self.update_var_params(init_var_params)

        enc_param_map = self.encode_feature(np.zeros(len(self.enc_param_names)))

        for var_p in self.var_params:
            var_p.set_value(var_param_map[var_p.name])
        for enc_p in self.enc_params:
            enc_p.set_value(enc_param_map[enc_p.name])
        return list(self.var_param_map.values())

    def update_var_params(self, updated):
        updated_dict = {}
        for i, p in enumerate(self.var_params):
            new_val = updated[i]
            updated_dict[p.name] = new_val
            p.set_value(new_val)
        self.var_param_map = updated_dict
        return updated_dict

    def encode_feature(self, feature):
        updated_dict = {}
        for i, p in enumerate(self.enc_params):
            new_val = feature[i]
            updated_dict[p.name] = new_val
            p.set_value(new_val)
        self.enc_param_map = updated_dict
        return updated_dict
