from collections import defaultdict

import numpy as np
import re

def poly_cleaning(string):
    string = re.sub("X0", "w", string)
    string = re.sub("X1", "x", string)
    string = re.sub("X2", "y", string)
    string = re.sub("X3", "z", string)
    return string

number_regex = re.compile("-?\d+\.\d+")

def poly_string_to_func(string):
    terms = string.split("+") 

    p = defaultdict(lambda: 0.)
    for term in terms:
        if term[0] == "-":
            coefficient = term[:5]
            monomial = term[5:]
        else:
            coefficient = term[:4]
            monomial = term[4:]
        p[monomial] = float(coefficient)

    def p_func(w, x, y, z):
        result = p[""] + p["w"] * w + p["x"] * x + p["y"] * y + p["z"] * z + p["wx"] * w * x + p["wy"] * w * y + p["wz"] * w * z + p["xw"] * x * w + p["xy"] * x * y + p["xz"] * x * z + p["yw"] * y * w + p["yx"] * y * x + p["yz"] * y * z + p["zw"] * z * w + p["zx"] * z * x + p["zy"] * z * y + p["w^2"] * (w ** 2) + p["x^2"] * (x ** 2) + p["y^2"] * (y ** 2) + p["z^2"] * (z ** 2)
        return result

    return p_func

def meta_string_to_dict(string):
    meta_mapping, mm_toe, example_toe, example = string.split(":")
    source, target = example.split("->")
    source_func = poly_string_to_func(poly_cleaning(source))
    target_func = poly_string_to_func(poly_cleaning(target))
    return {"meta_mapping": meta_mapping,
            "mm_toe": mm_toe,
            "example_toe": example_toe,
            "source": source,
            "target": target,
            "source_func": source_func,
            "target_func": target_func}

def compute_cross_loss(poly1, poly2, d=0.1):
    """The squared loss between the polynomials, approximated with d=0.05"""
    res = 0.
    num_points = 0
    for w in np.arange(-1., 1. + d, d):
        for x in np.arange(-1., 1. + d, d):
            for y in np.arange(-1., 1. + d, d):
                for z in np.arange(-1., 1. + d, d):
                    res += (poly1(w, x, y, z) - poly2(w, x, y, z)) ** 2
                    num_points += 1

    return res / num_points


if __name__ == "__main__":
    zero_func = poly_string_to_func("0.0")

    for run_i in range(5):
        with open("conditioned_vs_hyper_results/polynomials_results/run{}_meta_true_losses.csv".format(run_i), "r") as f:
            header = f.readline()[:-1]

        meta_points = header.split(", ")[1:]  # drop epoch

        meta_points = [meta_string_to_dict(x) for x in meta_points]

        with open("conditioned_vs_hyper_results/polynomials_results/run{}_meta_true_baselines.csv".format(run_i), "w") as fout:
            fout.write("meta_mapping, mapping_toe, base_task_toe, source, target, zeros_loss, unadapted_loss\n")
            for point in meta_points:
                zeros_loss = compute_cross_loss(point["target_func"], zero_func)
                unadapted_loss = compute_cross_loss(point["target_func"], point["source_func"])
                fout.write("{}, {}, {}, {}, {}, {}, {}\n".format(
                    point["meta_mapping"],
                    point["mm_toe"],
                    point["example_toe"],
                    point["source"],
                    point["target"],
                    zeros_loss,
                    unadapted_loss))

