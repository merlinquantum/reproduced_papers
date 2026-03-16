import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from utils.mappings import get_output_map, map_generator_output
from utils.pqc import ParametrizedQuantumCircuit


class ClassicalGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(2, 4, normalize=False), nn.Linear(4, int(np.prod((8, 8)))), nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 64)
        return img


# TODO: where the QuantumLayer of MerLin should be used
class PatchGenerator:
    def __init__(
        self,
        image_size,
        gen_count,
        gen_arch,
        input_state,
        pnr,
        lossy,
        remote_token=None,
        use_clements=False,
        sim=False,
    ):
        self.image_size = image_size
        self.gen_count = gen_count
        self.input_state = input_state
        self.generators = [
            ParametrizedQuantumCircuit(input_state.m, gen_arch, use_clements)
            for _ in range(gen_count)
        ]
        self.fake_data = None
        self.noise = None

        if remote_token is not None:
            if sim:
                proc = pcvl.RemoteProcessor("sim:ascella", token=remote_token)
            else:
                proc = pcvl.RemoteProcessor("qpu:ascella", token=remote_token)
            self.sample_count = 1000
        elif lossy:
            proc = pcvl.Processor(
                "SLOS",
                source=pcvl.Source(
                    losses=0.92,
                    emission_probability=1,
                    multiphoton_component=0,
                    indistinguishability=0.92,
                ),
            )
            self.sample_count = 1e5
        else:
            # sample_count = 1 for no sampling error
            self.sample_count = 1
            proc = pcvl.Processor("SLOS")

        proc.set_circuit(self.generators[0].circuit.copy())
        proc.with_input(self.input_state)
        proc.min_detected_photons_filter(self.input_state.n)
        if remote_token is not None:
            self.sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=1000000)
        else:
            self.sampler = pcvl.algorithm.Sampler(proc)

        for gen in self.generators:
            gen.init_params()

        self.output_map = get_output_map(
            self.generators[0].circuit, self.input_state, pnr, lossy
        )
        self.bin_count = np.max(list(self.output_map.values())) + 1

    def init_params(self):
        params = []
        for gen in self.generators:
            params.extend(list(gen.init_params()))
        return np.array(params)

    def update_var_params(self, params):
        param_count_per_gen = len(params) // self.gen_count
        for i, gen in enumerate(self.generators):
            gen.update_var_params(
                params[param_count_per_gen * i : param_count_per_gen * (i + 1)]
            )

    # build the sampler iteration list
    def get_iteration_list(self):
        iteration_list = []
        for z in self.noise:
            for gen in self.generators:
                gen.encode_feature(z)
                params = gen.var_param_map.copy()
                params.update(gen.enc_param_map)
                iteration_list.append({"circuit_params": params})
        return iteration_list

    def generate(self, noise=None, it_list=None):
        if noise is not None:
            self.noise = noise

        if it_list is None:
            iteration_list = self.get_iteration_list()
        else:
            iteration_list = it_list.copy()

        # flush the previous run and sample using the iteration list
        try:
            self.sampler._iterator = []
            self.sampler.add_iteration_list(iteration_list)
            if self.sample_count == 1:
                result_list = self.sampler.probs()["results_list"]
            else:
                result_list = self.sampler.sample_count(self.sample_count)[
                    "results_list"
                ]
            result_list = np.array(result_list).reshape((-1, self.gen_count))
        except Exception as exc:
            print(exc)
            return self.fake_data

        # build fake data based on sampling results
        out_map = self.output_map
        fake_data = []
        for noise_item in result_list:
            fake_data_sample = []
            for gen_item in noise_item:
                res = gen_item["results"]

                gen_out = np.zeros(self.bin_count)
                total_count = 0
                for key in res.keys():
                    try:
                        gen_out[out_map[key]] += res[key]
                        total_count += res[key]
                    except Exception:
                        pass
                # print(np.sum(gen_out / self.sample_count), np.sum(gen_out / total_count))
                if total_count > 0:
                    gen_out /= total_count
                out_modes = map_generator_output(
                    gen_out, self.image_size * self.image_size // self.gen_count
                )
                max_mode = np.max(out_modes)
                if max_mode > 0:
                    out_modes = out_modes / max_mode
                fake_data_sample.extend(out_modes)

            fake_data.append(fake_data_sample)
        fake_data = torch.FloatTensor(fake_data)

        self.fake_data = fake_data
        return fake_data
