import time
import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax
from collections import deque
import networkx as nx


from viz import create_backend
from src.cppn_utils import activation_fns, build_coordinate_grid, define_coordinate_inputs

class CPPN:
    def __init__(self, output_path, params):
        self.output_path = output_path
        self.params = params

        # General config
        self.default_res = 1024
        self.high_res = 2048
        self.factor = params['factor']
        self.debug = params.get('debug', False)

        # Architecture
        self.n_inputs = 6
        self.n_hidden = 10
        self.n_outputs = 3
        self.n_nodes = self.n_inputs + self.n_hidden + self.n_outputs

        self.input_ids = list(range(0, self.n_inputs))
        self.middle_ids = list(range(self.n_inputs, self.n_inputs + self.n_hidden))
        self.output_ids = list(range(self.n_inputs + self.n_hidden, self.n_nodes))

        # Parameters
        self.adj_matrix = np.zeros((self.n_nodes, self.n_nodes))
        self.biases = jnp.zeros(self.n_hidden)
        self.slopes = jnp.zeros(self.n_hidden)
        self.weights = jnp.zeros((self.n_nodes, self.n_nodes))
        self.activation_ids = jnp.zeros(self.n_hidden, dtype=int)

        # Output transformation (RGB or HSL mode)
        self.output_slopes = jnp.ones(3)   # scaling for output channels
        self.output_biases = jnp.zeros(3)  # sigmoid or linear bias
        self.output_modes = jnp.zeros(3, dtype=int)  # 0=sigmoid RGB, 1=clipped HSL

        # Input configuration (for inputs 1–6)
        self.input_function_ids = [0, 1, 2, 3, 4, 5]  # Index into input basis functions
        self.input_function_keys = ['dist', 'angle', 'radial_wave', 'grid', 'r_mod', 'spiral', 'ripple', 'star']
        self.input_zooms = jnp.ones(self.n_inputs)  # Per-input zoom
        self.input_biases = jnp.zeros(self.n_inputs)  # Per-input bias
        self.input_inverted = jnp.zeros(self.n_inputs, dtype=int)  # Per-input invertion

        # Node masking
        self.node_active = jnp.ones(self.n_inputs + self.n_hidden)
        self.cv_override = jnp.zeros(self.n_nodes)
        self.inputs_1 = np.full(self.n_nodes, -1, dtype=int)  # track what gets connected on inputs

        # Activations
        self.activations = activation_fns

        # Precomputed coordinate grids
        X_low, Y_low = build_coordinate_grid(self.default_res, self.factor)
        X_high, Y_high = build_coordinate_grid(self.high_res, self.factor)

        self.input_maps = {self.default_res: define_coordinate_inputs(X_low, Y_low),
                           self.high_res: define_coordinate_inputs(X_high, Y_high)}

        # Runtime
        self.needs_update = True
        self.first = True

        # Placeholder for compiled forward pass
        self.jitted_process_network = None
        self._init_jitted_functions()
        self.sample_network()

    def _init_jitted_functions(self):
        n_inputs = self.n_inputs
        n_hidden = self.n_hidden
        n_outputs = self.n_outputs
        n_nodes = self.n_nodes
        activations = self.activations

        @jit
        def process_inputs(basis, state):
            def process_one(i):
                val = basis[state['function_ids'][i]]
                val = (val / state['zooms'][i]) + state['biases_input'][i]
                val = jnp.where(state['inversions'][i], -val, val)
                return val * state['node_active'][i]

            inputs = jnp.stack([process_one(i) for i in range(n_inputs)])
            return inputs.reshape(n_inputs, -1)

        @jit
        def apply_activation(x, activation_id):
            return lax.switch(activation_id, activations, x)

        @jit
        def process_hidden_node(weights, x, bias, slope, activation_id, active_flag):
            net = jnp.dot(weights, x)
            return apply_activation((net + bias) * slope, activation_id) * active_flag

        @jit
        def process_output_node(weights, x, bias, slope):
            net = jnp.dot(weights, x)
            return (net + bias) * slope

        @jit
        def hsl_to_rgb(h, s, l):
            def f(n):
                k = (n + h * 6.0) % 6.0
                a = s * jnp.minimum(l, 1.0 - l)
                return l - a * jnp.maximum(-1.0, jnp.minimum(jnp.minimum(k - 3.0, 9.0 - k), 1.0))

            return jnp.stack([f(0.0), f(2.0), f(4.0)], axis=0)

        @jit
        def run_network(basis, state, cyclic_start):
            weights = state['weights'] * state['adj_matrix']
            inputs_flat = process_inputs(basis, state)

            x = jnp.zeros((n_nodes, inputs_flat.shape[1]))
            x = x.at[:n_inputs].set(inputs_flat)

            def hidden_cycle(i_cycle, x):
                start = lax.select(i_cycle == 0, n_inputs, cyclic_start)
                end = n_inputs + n_hidden

                def hidden_loop(i, x):
                    h_idx = i - n_inputs
                    return x.at[i].set(
                        process_hidden_node(
                            weights[:, i],
                            x,
                            state['biases'][i],
                            state['slopes'][i],
                            state['activation_ids'][i],
                            state['node_active'][h_idx]
                        )
                    )

                return lax.fori_loop(start, end, hidden_loop, x)

            x = lax.fori_loop(0, 2, hidden_cycle, x)

            def output_loop(i, outputs):
                idx = n_inputs + n_hidden + i
                val = process_output_node(
                    weights[:, idx],
                    x,
                    state['output_biases'][i],
                    state['output_slopes'][i]
                )
                return outputs.at[i].set(val)

            outputs = jnp.zeros((n_outputs, inputs_flat.shape[1]))
            outputs = lax.fori_loop(0, n_outputs, output_loop, outputs)

            out = lax.cond(
                jnp.any(state['output_modes'] > 0),
                lambda _: hsl_to_rgb(outputs[0], outputs[1], outputs[2]),
                lambda _: outputs,
                operand=None
            )

            H, W = basis.shape[1], basis.shape[2]
            return out.reshape(3, H, W).transpose(1, 2, 0)

        self.jitted_process_network = run_network

    def update(self, res=None):
        if res is None:
            res = self.default_res

        t0 = time.time()
        state = {
            'weights': self.weights,
            'adj_matrix': self.adj_matrix,
            'function_ids': jnp.array(self.input_function_ids),
            'zooms': self.input_zooms,
            'biases_input': self.input_biases,
            'inversions': self.input_inverted,
            'node_active': self.node_active[:self.n_inputs],
            'biases': self.biases,
            'slopes': self.slopes,
            'activation_ids': self.activation_ids,
            'output_biases': self.output_biases,
            'output_slopes': self.output_slopes,
            'output_modes': self.output_modes,
        }

        self.current_image = self.jitted_process_network(
            self.input_maps[res],
            state,
            self.cyclic_start
        )



        self.needs_update = False

        if self.debug:
            print(f"[CPPN] Updated at res={res} in {time.time() - t0:.3f}s")

        return self.current_image

    def validate_graph(self):
        """
        - Updates self.cyclic_start based on current adj_matrix
        - Returns True if there is at least one input→output path, False otherwise
        """
        ordered = list(self.input_ids)
        available = set(self.middle_ids)

        while available:
            added = False
            for node in sorted(available):
                inputs = np.where(self.adj_matrix[:, node])[0]
                if all(i in ordered for i in inputs):
                    ordered.append(node)
                    available.remove(node)
                    added = True
                    break
            if not added:
                break

        self.cyclic_start = len(ordered)

        # Check input→output path
        G = nx.DiGraph(self.adj_matrix)
        for inp in self.input_ids:
            for out in self.output_ids:
                if nx.has_path(G, inp, out):
                    return True

        return False

    def sample_network(self):
        while True:
            # Step 1: Sample forward connections only
            self.adj_matrix = np.zeros((self.n_nodes, self.n_nodes))
            for i in range(self.n_inputs + self.n_hidden):
                for j in range(self.n_inputs, self.n_nodes):
                    if j > i and np.random.rand() < 0.8:
                        self.adj_matrix[i, j] = 1

            # Step 2: Check for valid paths and compute cyclic start
            if self.validate_graph():
                break

        # Step 3: Random weights only on existing connections
        self.weights = jnp.array(
            (np.random.randn(self.n_nodes, self.n_nodes) * self.adj_matrix).astype(np.float32)
        )

        # Step 4: Convert adjacency to JAX array
        self.adj_matrix = jnp.array(self.adj_matrix)

        self.activation_ids = jnp.array(np.random.randint(0, len(self.activations), size=self.n_hidden), dtype=jnp.int32)

        self.needs_update = True
        print(f"[CPPN] Sampled new network with cyclic start at {self.cyclic_start}")



if __name__ == '__main__':
    RES = 1024
    FACTOR = 16 / 9

    vis = create_backend('pygame')
    vis.initialize(
        render_width=int(RES * FACTOR),
        render_height=RES,
        window_scale=1  # Adjust this to make window bigger/smaller
    )

    params = dict(debug=True, res=RES, factor=FACTOR)

    cppn = CPPN(output_path='...', params=params)
    cppn.sample_network()
    img = cppn.update()
    vis.update(img)

    for _ in range(10):
        time.sleep(1)
        cppn.sample_network()
        img = cppn.update()
        vis.update(img)