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
    print(p)

    def p_func(w, x, y, z):
        result = p[""] + p["w"] * w + p["x"] * x + p["y"] * y + p["z"] * z + p["wx"] * w * x + p["wy"] * w * y + p["wz"] * w * z + p["xw"] * x * w + p["xy"] * x * y + p["xz"] * x * z + p["yw"] * y * w + p["yx"] * y * x + p["yz"] * y * z + p["zw"] * z * w + p["zx"] * z * x + p["zy"] * z * y + p["w^2"] * (w ** 2) + p["x^2"] * (x ** 2) + p["y^2"] * (y ** 2) + p["z^2"] * (z ** 2)
        return result

    return p_func

def meta_string_to_dict(string):
    example = string.split(":")[-1] 
    source, target = example.split("->")
    source = poly_string_to_func(poly_cleaning(source))
    target = poly_string_to_func(poly_cleaning(target))
    return {"string": string,
            "source": source,
            "target": target}


with open("conditioned_vs_hyper_results/polynomials_results/run0_meta_true_losses.csv", "r") as f:
    header = f.readline()[:-1]

meta_points = header.split(", ")[1:]  # drop epoch

meta_points = [meta_string_to_dict(x) for x in meta_points]


if __name__ == "__main__":
    test_poly = "4.25+2.78X0+-1.10X1+0.45X3+0.22X0^2+0.66X0X3"
    test_poly = poly_cleaning(test_poly)
    p = poly_string_to_func(test_poly)
    print(p(0, 0, 0, 0))
    print(p(1., 0, 0, 0))
    print(p(1, 0, 0, 1))

