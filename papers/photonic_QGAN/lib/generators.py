import merlin as ML
import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.mappings import get_output_map, map_generator_output
from utils.pqc import ParametrizedQuantumCircuit


class ClassicalGenerator(nn.Module):
    def __init__(self, noise_dim=2, image_size=8, hidden_dim=64):
        super().__init__()
        self.noise_dim = int(noise_dim)
        self.image_size = int(image_size)
        self.hidden_dim = int(hidden_dim)
        output_dim = int(np.prod((self.image_size, self.image_size)))

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.noise_dim, self.hidden_dim, normalize=False),
            nn.Linear(self.hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.image_size * self.image_size)
        return img


class PatchGenerator(nn.Module):
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
        super().__init__()
        self.image_size = image_size
        self.gen_count = gen_count
        self.input_state = input_state

        # Here I have replaced the list of ParametrizedQuantumCircuit
        # By a list of quantum layers based on these circuits
        self.models = nn.ModuleList()
        for _ in range(gen_count):
            pcvl_circuit = ParametrizedQuantumCircuit(
                len(input_state), gen_arch, use_clements
            )
            circuit_var_params = pcvl_circuit.var_param_names
            circuit_enc_params = pcvl_circuit.enc_param_names
            num_enc_params = len(circuit_enc_params)

            layer = ML.QuantumLayer(
                input_size=num_enc_params,
                circuit=pcvl_circuit.circuit,
                input_parameters=circuit_enc_params,
                trainable_parameters=circuit_var_params,
                input_state=self.input_state,
                measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
                computation_space=ML.ComputationSpace.FOCK,
            )

            self.models.append(layer)

        # Define mapping
        self.output_keys = self.models[0].output_keys
        rev_map = {}
        possible_outputs = []

        def state_to_int(state, pnr):
            m = len(state)
            res = 0
            for i in range(m):
                if pnr:
                    res += state[i] * (m + 1) ** (m - i)
                elif state[i] != 0:
                    res += 2 ** (m - i)
            return res

        # Mirror get_output_map: with threshold detectors (non-pnr) and lossy source,
        # only keep states where every mode has count < 2.
        if pnr or not lossy:
            possible_state_keys = self.output_keys
        else:
            possible_state_keys = [
                key for key in self.output_keys if all(i < 2 for i in key)
            ]

        for key in possible_state_keys:
            int_state = state_to_int(key, pnr)
            if int_state in rev_map.keys():
                rev_map[int_state].append(key)
            else:
                rev_map[int_state] = [key]
            if int_state not in possible_outputs:
                possible_outputs.append(int_state)

        self.output_map = {}
        for index, int_state in enumerate(sorted(possible_outputs)):
            for basic_state in rev_map[int_state]:
                self.output_map[basic_state] = index

        self.bin_count = np.max(list(self.output_map.values())) + 1
        self.expected_size = self.image_size * self.image_size // self.gen_count

        # Precompute per-column index tensors for dist_to_image.
        # When lossy and not pnr, output_map covers only a subset of output_keys;
        # restrict to those columns so no KeyError is raised.
        self._mapped_col_indices = torch.tensor(
            [i for i, k in enumerate(self.output_keys) if k in self.output_map],
            dtype=torch.long,
        )
        self._idx_cpu = torch.tensor(
            [self.output_map[k] for k in self.output_keys if k in self.output_map],
            dtype=torch.long,
        )

    def dist_to_image(self, raw_results_list):
        patches = []
        B = None
        K = len(self.output_keys)

        for res in raw_results_list:
            # res: [B, K]
            if res.numel() == 0:
                continue

            # for each of the generators results, rearrange prob distribution according to get_output_map
            if res.shape[1] != K:
                raise ValueError(
                    f"res has K={res.shape[1]} cols but len(output_keys)={K}"
                )

            if B is None:
                B = res.shape[0]
            elif res.shape[0] != B:
                raise ValueError(
                    f"Batch size mismatch: got {res.shape[0]} vs expected {B}"
                )

            device = res.device
            dtype = res.dtype
            idx = self._idx_cpu.to(device=device)
            col_idx = self._mapped_col_indices.to(device=device)

            # Restrict to columns present in output_map (filtered subset when lossy and not pnr)
            res_mapped = res.index_select(1, col_idx)  # [B, len(mapped)]

            gen_out = torch.zeros((B, self.bin_count), device=device, dtype=dtype)
            gen_out.index_add_(1, idx, res_mapped)

            # Normalize by the probability mass that landed in mapped (kept) bins.
            # In lossy/non-PNR mode this can be 0 for samples where all probability
            # fell into filtered-out multi-photon states; clamp ensures no NaN/inf
            # and the result is a zero distribution for those samples.
            total_count = res_mapped.sum(dim=1, keepdim=True)  # [B, 1]
            gen_out = gen_out / total_count.clamp(min=1e-8)

            # map to the right number of pixels with map_generator_output
            gen_out_len = gen_out.shape[1]
            expected_len = self.expected_size

            if gen_out_len > expected_len:
                surplus_half = (gen_out_len - expected_len) // 2
                img_gen = gen_out[:, surplus_half : surplus_half + expected_len]
            else:
                left = (expected_len - gen_out_len) // 2
                right = expected_len - gen_out_len - left
                img_gen = F.pad(gen_out, (left, right))

            # Normalize patch by its max (avoid divide-by-zero)
            mx = img_gen.max(dim=1, keepdim=True).values
            img_gen = img_gen / (mx + 1e-8)

            # Concatenate into one long "img_patch" tensor
            patches.append(img_gen)

        # Concatenate into one patch
        if len(patches) == 0:
            return torch.empty(0)

        return torch.cat(patches, dim=1)

    def forward(self, z):
        # Get results from each generator which is a quantum layer
        raw_results_list = [m(z) for m in self.models]

        # Map to image
        img = self.dist_to_image(raw_results_list)

        return img


class PatchGeneratorLegacy:
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
        # Do the mapping to get an image
        # what the mapping does:
        # for 1 item in a batch, get results for all generators in the patchgenerator
        # for each of them, rearrange prob distribution according to get_output_map
        # then map to the right number of pixels with map_generator_output
        # then concatenate all in one array --> that's one image
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
            # There are (n_batch, n_generators) items in the whole list
            # We create one data sample per batch item, based on all generators
            fake_data_sample = []
            for gen_item in noise_item:
                res = gen_item["results"]

                gen_out = np.zeros(self.bin_count)
                total_count = 0
                for key in res.keys():
                    try:
                        gen_out[out_map[key]] += res[key]
                        total_count += res[key]
                    except KeyError:
                        continue
                # print(np.sum(gen_out / self.sample_count), np.sum(gen_out / total_count))
                gen_out /= total_count
                # The probabilities are re-ordered according to the mapping
                # gen_out is a np array of the re-ordered probabilities of size bin_count
                # map_generator_output then maps gen_out to the right number of pixels
                out_modes = map_generator_output(
                    gen_out, self.image_size * self.image_size // self.gen_count
                )
                # add linearly to fake data sample
                mx = np.max(out_modes)
                fake_data_sample.extend(out_modes / mx if mx > 0 else out_modes)

            fake_data.append(fake_data_sample)
        fake_data = torch.FloatTensor(fake_data)

        self.fake_data = fake_data
        return fake_data
