import numpy as np
import perceval as pcvl


def state_to_int(state, pnr):
    m = state.m
    res = 0
    for i in range(m):
        # pnr
        if pnr:
            res += state[i] * (m + 1) ** (m - i)
        # threshold
        elif state[i] != 0:
            res += 2 ** (m - i)
    return res


# generates a mapping dictionary from output Fock states to integers
def get_output_map(circuit, input_state, pnr=True, lossy=False):
    proc = pcvl.Processor("SLOS")
    proc.set_circuit(circuit)
    proc.with_input(input_state)
    sampler = pcvl.algorithm.Sampler(proc)

    rev_map = {}
    possible_outputs = []

    all_states = sampler.probs()["results"].keys()
    if pnr or not lossy:
        possible_states_list = list(all_states)
    else:
        possible_states_list = [key for key in all_states if all(i < 2 for i in key)]

    for key in possible_states_list:
        int_state = state_to_int(key, pnr)
        # while int_state in rev_map:
        #     int_state += 1
        if int_state in rev_map.keys():
            rev_map[int_state].append(key)
        else:
            rev_map[int_state] = [key]
        if int_state not in possible_outputs:
            possible_outputs.append(int_state)

    out_map = {}
    for index, int_state in enumerate(sorted(possible_outputs)):
        for basic_state in rev_map[int_state]:
            out_map[basic_state] = index
    return out_map


# maps the output of a generator circuit to image pixels
def map_generator_output(gen_out, expected_len):
    gen_out_len = len(gen_out)
    surplus_half = np.abs((gen_out_len - expected_len) // 2)

    if gen_out_len > expected_len:
        return gen_out[surplus_half : surplus_half + expected_len]
    else:
        ret = np.zeros(expected_len)
        ret[surplus_half : surplus_half + gen_out_len] = gen_out
        return ret
