from itertools import permutations

from copy import deepcopy

import numpy as np
import tensorflow as tf

from HoMM import HoMM_model
from HoMM.configs import default_run_config, default_architecture_config
import polynomials

run_config = default_run_config.default_run_config
run_config.update({
    "output_dir": "/mnt/fs3/lampinen/polynomials/polynomials_mapped_rep_results/",
    "run_offset": 0,
    "num_runs": 4,
    
    "num_base_train_tasks": 60, # prior to meta-augmentation
    "num_base_eval_tasks": 40, # prior to meta-augmentation

    "num_variables": 4,
    "max_degree": 2,
    "poly_coeff_sd": 2.5,
    "point_val_range": 1,

    "num_epochs": 5000,
#    "num_optimization_epochs": 250,

    "meta_add_vals": [-3, -1, 1, 3],
    "meta_mult_vals": [-3, -1, 3],
    "new_meta_tasks": [],
    "new_meta_mappings": ["add_%f" % 2., "add_%f" % -2., "mult_%f" % 2., "mult_%f" % -2.],
})

architecture_config = default_architecture_config.default_architecture_config
#architecture_config.update({
#    "train_drop_prob": 0.5,
#})
if False:  # enable for tcnh
    architecture_config.update({
        "task_conditioned_not_hyper": True,
    })
    run_config.update({
        "output_dir": run_config["output_dir"] + "tcnh/", 
    })


if False:  # enable for nonhomiconic, after ensuring HoMM is on correct branch 
    architecture_config.update({
        "nonhomoiconic": True,
    })
    run_config.update({
        "output_dir": run_config["output_dir"] + "nonhomoiconic/", 
    })

if False:  # enable for weight norm 
    architecture_config.update({
        "F_weight_normalization": True,
    })
    run_config.update({
        "output_dir": run_config["output_dir"][:-1] + "_weight_norm/", 
    })
    if True:  # Weight norm properly 
        architecture_config.update({
            "F_wn_strategy": "standard",
        })
        run_config.update({
            "output_dir": run_config["output_dir"][:-1] + "_strategy_standard/", 
        })
    if False:  # Weight norm last only 
        architecture_config.update({
            "F_wn_strategy": "unit_until_last",
        })
        run_config.update({
            "output_dir": run_config["output_dir"][:-1] + "_strategy_unit_until_last/", 
        })

class poly_HoMM_model(HoMM_model.HoMM_model):
    def __init__(self, run_config=None):
        super(poly_HoMM_model, self).__init__(
            architecture_config=architecture_config, run_config=run_config)

    def _pre_build_calls(self):
        # set up the base tasks
        poly_fam = polynomials.polynomial_family(self.run_config["num_variables"], self.run_config["max_degree"])
        self.run_config["variables"] = poly_fam.variables

        self.base_train_tasks = [poly_fam.sample_polynomial(coefficient_sd=self.run_config["poly_coeff_sd"]) for _ in range(self.run_config["num_base_train_tasks"])]
        self.base_eval_tasks = [poly_fam.sample_polynomial(coefficient_sd=self.run_config["poly_coeff_sd"]) for _ in range(self.run_config["num_base_eval_tasks"])]

        # set up the meta tasks

        self.meta_class_train_tasks = ["is_constant_polynomial"] + ["is_intercept_nonzero"] + ["is_%s_relevant" % var for var in self.run_config["variables"]]
        self.meta_class_eval_tasks = [] 

        self.meta_map_train_tasks = ["square"] + ["add_%f" % c for c in self.run_config["meta_add_vals"]] + ["mult_%f" % c for c in self.run_config["meta_mult_vals"]]
        self.meta_map_eval_tasks = self.run_config["new_meta_mappings"] 

        permutation_mappings = ["permute_" + "".join([str(x) for x in p]) for p in permutations(range(self.run_config["num_variables"]))]
        np.random.seed(0)
        np.random.shuffle(permutation_mappings)
        self.meta_map_train_tasks += permutation_mappings[:len(permutation_mappings)//2]
        self.meta_map_eval_tasks += permutation_mappings[len(permutation_mappings)//2:]

        # set up language

        #vocab = ['PAD'] + [str(x) for x in range(10)] + [".", "+", "-", "^"] + poly_fam.variables
        #vocab_to_int = dict(zip(vocab, range(len(vocab))))
        #
        #self.run_config["vocab"] = vocab

        # set up the meta pairings 
        self.meta_pairings, implied_tasks_train_tasks, implied_tasks_eval_tasks = polynomials.get_meta_pairings(
            base_train_tasks=self.base_train_tasks,
            base_eval_tasks=self.base_eval_tasks,
            meta_class_train_tasks=self.meta_class_train_tasks,
            meta_class_eval_tasks=self.meta_class_eval_tasks,
            meta_map_train_tasks=self.meta_map_train_tasks,
            meta_map_eval_tasks=self.meta_map_eval_tasks) 

        # for these experiments, we want every eval task to be a meta-mapping
        # of a trained task, so we just move all base_eval_tasks to train, and replace them with implied.
        self.base_train_tasks += self.base_eval_tasks 
        self.base_eval_tasks = implied_tasks_eval_tasks

        # add the base tasks implied by the mappings
        self.base_train_tasks += implied_tasks_train_tasks

    def fill_buffers(self, num_data_points=1):
        """Add new "experiences" to memory buffers."""
        this_tasks = self.base_train_tasks + self.base_eval_tasks 
        for t in this_tasks:
            buff = self.memory_buffers[polynomials.stringify_polynomial(t)]
            x_data = np.zeros([num_data_points] + self.architecture_config["input_shape"])
            y_data = np.zeros([num_data_points] + self.architecture_config["output_shape"])
            for point_i in range(num_data_points):
                point = t.family.sample_point(val_range=self.run_config["point_val_range"])
                x_data[point_i, :] = point
                y_data[point_i, :] = t.evaluate(point)
            buff.insert(x_data, y_data)

    def intify_task(self, task):
        if task == "square":
            return [vocab_to_int[task]]
        elif task[:3] == "add":
            val = str(int(round(float(task[4:]))))
            return [vocab_to_int["add"]] + [vocab_to_int[x] for x in val]
        elif task[:4] == "mult":
            val = str(int(round(float(task[5:]))))
            return [vocab_to_int["mult"]] + [vocab_to_int[x] for x in val]
        elif task[:7] == "permute":
            val = task[8:]
            return [vocab_to_int["permute"]] + [vocab_to_int[x] for x in val]
        elif task[:3] == "is_":
            if task[3] == "X":
                return [vocab_to_int[x] for x in task.split('_')] 
            else:
                return [vocab_to_int["is"], vocab_to_int[task[3:]]]
        else:
            raise ValueError("Unrecognized meta task: %s" % task)

    def save_base_data_representations(self, filename):
        input_range = np.arange(-1., 1.1, 0.2)
        num_examples = 4 * len(input_range) + 6 * len(input_range)**2
        this_inputs = np.zeros([num_examples, 4])
        i = 0
        for var_i in range(4):
            # each var independently
            for val in input_range:
                curr_input = np.zeros([4])
                curr_input[var_i] = val
                this_inputs[i] = curr_input
                i += 1

            # and pairs
            for val in input_range:
                for var_j in range(var_i + 1, 4):
                    for val2 in input_range:
                        curr_input = np.zeros([4])
                        curr_input[var_i] = val
                        curr_input[var_j] = val2
                        this_inputs[i] = curr_input
                        i += 1

        feed_dict = {self.base_input_ph: this_inputs,
                     self.keep_prob_ph: 1.}
        this_results = self.sess.run(self.processed_input, feed_dict=feed_dict)

        with open(filename, "w") as fout:
            num_dims = this_results.shape[1]
            fout.write("X0, X1, X2, X3, " + ", ".join(["dim{}".format(x) for x in range(num_dims)]) + "\n") 
            format_string = ", ".join(["%f"] * (num_dims + 4)) + "\n"
            for result_i in range(num_examples):
                fout.write(format_string % (tuple(this_inputs[result_i, :]) + tuple(this_results[result_i, :])))



## running stuff
for run_i in range(run_config["run_offset"], run_config["run_offset"] + run_config["num_runs"]):
    np.random.seed(run_i)
    tf.set_random_seed(run_i)
    run_config["this_run"] = run_i

    model = poly_HoMM_model(run_config=run_config)
    #model.save_meta_pairings()
    #model.run_training()
    #model.save_parameters(model.filename_prefix + "final_checkpoint")
    model.restore_parameters(model.filename_prefix + "final_checkpoint")
    model.save_base_data_representations(model.filename_prefix + "input_representations.csv")
    #model.run_varied_meta_batch_eval()
    #model.save_task_embeddings(model.filename_prefix + "task_representations.csv")
    #model.save_metamapped_task_embeddings(model.filename_prefix + "task_representations_", save_eval=True)
    #model.guess_embeddings_and_optimize(num_optimization_epochs=run_config["num_optimization_epochs"], eval_every=2, random_init_scale=0.1)

    tf.reset_default_graph()
