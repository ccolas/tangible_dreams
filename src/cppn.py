import time

import PIL.Image
import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax
from collections import deque
import networkx as nx
from PIL import Image
import pickle
import cv2
import matplotlib.pyplot as plt


from viz import create_backend
from src.cppn_utils import activation_fns, build_coordinate_grid, input_functions, zoom_mapping, bias_mapping

class CPPN:
    def __init__(self, output_path, params):
        self.output_path = output_path
        self.params = params
        self.t_start = time.time()
        self.last_time = 0
        self.update_period = 0.03
        self.reactive = False
        self.cam_input = False

        # General config
        self.default_res = 1024
        self.high_res = 2048
        self.factor = params['factor']
        self.debug = params.get('debug', False)

        # Architecture
        self.n_inputs = 6
        self.n_hidden = params['n_middle_nodes']
        self.n_outputs = 3
        self.n_nodes = self.n_inputs + self.n_hidden + self.n_outputs
        self.n_cycles = 3

        self.input_ids = list(range(0, self.n_inputs))
        self.middle_ids = list(range(self.n_inputs, self.n_inputs + self.n_hidden))
        self.output_ids = list(range(self.n_inputs + self.n_hidden, self.n_nodes))

        # Parameters
        self.adj_matrix = jnp.zeros((self.n_nodes, self.n_nodes))
        self.biases = jnp.zeros(self.n_hidden)
        self.slopes = jnp.ones(self.n_hidden)
        self.slope_mods = jnp.zeros(self.n_hidden)
        self.weights = jnp.zeros((self.n_nodes, self.n_nodes))
        self.weight_mods = jnp.zeros((self.n_nodes, self.n_nodes))
        self.activation_ids = jnp.zeros(self.n_hidden, dtype=int)

        # Output transformation (RGB or HSL mode)
        self.output_slopes = jnp.ones(3)   # scaling for output channels
        self.output_slope_mods = jnp.zeros(3)   # scaling for output channels
        self.output_biases = jnp.zeros(3)  # sigmoid or linear bias
        self.output_modes = jnp.zeros(3, dtype=int)  # 0=sigmoid RGB, 1=clipped HSL

        # Input configuration (for inputs 1–6)
        self.input_function_ids = jnp.array([0, 1, 2, 3, 4, 5])  # Index into input basis functions
        self.input_params1 = jnp.full(self.n_inputs, 512)  # zoom → default 1
        self.input_params2 = jnp.full(self.n_inputs, 512)  # bias → default 0
        # self.input_inverted = jnp.zeros(self.n_inputs, dtype=int)  # Per-input invertion

        # Node masking
        self.node_active = jnp.ones(self.n_inputs + self.n_hidden)
        self.cv_override = jnp.zeros(self.n_nodes)
        self.inputs_nodes_record = np.full((self.n_nodes, 3), -1, dtype=int)  # track what gets connected on inputs

        # Activations
        self.activations = list(activation_fns.values())
        self.activations_names = list(activation_fns.keys())

        # Precomputed coordinate grids
        self.input_functions = input_functions
        self.coords = {self.default_res: build_coordinate_grid(self.default_res, self.factor),
                       self.high_res: build_coordinate_grid(self.high_res, self.factor)}
        self.img_shapes = {self.default_res: self.coords[self.default_res][0].shape,
                           self.high_res: self.coords[self.high_res][0].shape}

        # Camera input
        self.cam = cv2.VideoCapture(0)  # webcam index 0
        self.cam_img = np.zeros(self.img_shapes[self.default_res], dtype=np.float32)  # normalized grayscale

        # Runtime
        self.needs_update = True
        self.first = True

        # Placeholder for compiled forward pass
        self.jitted_process_network = None
        self._init_jitted_functions()
        self.sample_network()

        self.set_pressed = False
        if params.get('load_from'):
            self.set_state(state_path=params['load_from'])

    def update_cam(self, res=None):
        if res is None:
            res = self.default_res
        # TODO: find something better here
        if self.cam_input:
            time = self.time
            if time - self.last_time > self.update_period:
                target_shape = self.img_shapes[res]
                ret, frame = self.cam.read()
                if not ret:
                    return

                # Flip for mirror view
                gray = cv2.flip(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2GRAY), 1)

                # Denoise with Gaussian blur
                k = 11
                sigma = 1
                blurred = cv2.GaussianBlur(gray, (k, k), sigma)
                blurred = cv2.resize(blurred, (target_shape[1], target_shape[0]))  # W x H

                grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
                self.cam_img = grad_mag / (grad_mag.max() + 1e-6)

                # self.cam_img = discretized.astype(np.float32)
                self.needs_update = True

    # def reactive_update(self):
    #     if self.reactive:
    #         time = self.time
    #         if time - self.last_time > self.update_period:
    #             self.last_time = time
    #
    #             # Toroidal signal with two incommensurate frequencies
    #             alpha, beta = 1.5, 2.3
    #             signal1 = np.sin(alpha * time)
    #             signal2 = np.cos(beta * time)
    #
    #             # Combine or use independently
    #             value = signal1 * 0.5 + signal2 * 0.5
    #
    #             # print(value)
    #             self.slope_mods = self.slope_mods.at[0].set(value)
    #             self.needs_update = True
    def reactive_update(self):
        if self.reactive:
            time = self.time
            if time - self.last_time > self.update_period:
                self.last_time = time

                # Smooth, pseudo-chaotic looping
                a, b = 2.0, 3.1
                value = np.sin(a * time + np.sin(b * time))  # in [-1, 1]

                print(value)
                # self.weights = self.weights.at[7, 9].set(value)
                self.slope_mods = self.slope_mods.at[0].set(value)
                self.needs_update = True



    def update(self, res=None):
        if res is None:
            res = self.default_res
        if res != self.default_res:
            self.update_cam(res)

        coord_x, coord_y = self.coords[res]

        # t0 = time.time()
        state = {
            'weights': self.weights,
            'weight_mods': self.weight_mods,
            'adj_matrix': self.adj_matrix,
            'input_function_ids': jnp.array(self.input_function_ids),
            'input_params1': self.input_params1,
            'input_params2': self.input_params2,
            'node_active': self.node_active,
            'biases': self.biases,
            'slopes': self.slopes,
            'slope_mods': self.slope_mods,
            'activation_ids': self.activation_ids,
            'output_biases': self.output_biases,
            'output_slopes': self.output_slopes,
            'output_slope_mods': self.output_slope_mods,
            'output_modes': self.output_modes,
            'cam_img': self.cam_img,
            'cv_override': self.cv_override,
        }
        self.current_image = self.jitted_process_network(coord_x, coord_y, state, self.cyclic_start)
        self.needs_update = False
        return self.current_image

    def _init_jitted_functions(self):
        n_inputs = self.n_inputs
        n_hidden = self.n_hidden
        n_outputs = self.n_outputs
        n_nodes = self.n_nodes
        activations = self.activations

        def define_basis_fn():
            return lambda fn_id, x, y, p1, p2: lax.switch(fn_id, self.input_functions, x, y, p1, p2)

        basis_fn = define_basis_fn()

        @jit
        def process_inputs(state, x, y, cam_img):
            def compute_input(i):
                general_zoom_x = zoom_mapping(state['input_params1'][0])  # remap values
                general_zoom_y = zoom_mapping(state['input_params1'][1])
                general_bias_x = bias_mapping(state['input_params2'][0])
                general_bias_y = bias_mapping(state['input_params2'][1])

                p1 = state['input_params1'][i]  # in 0-1023
                p2 = state['input_params2'][i]  # in 0-1023
                f_id = state['input_function_ids'][i]

                def coords_(x, y):
                    x_i = (x + general_bias_x) / general_zoom_x
                    y_i = (y + general_bias_y) / general_zoom_y
                    return x_i, y_i

                x_i, y_i = coords_(x, y)

                val = lax.cond(
                    f_id == -1,
                    lambda _: cam_img,  # match output shape of val
                    lambda _: basis_fn(f_id, x_i, y_i, p1, p2),
                    operand=None
                )
                return val * state['node_active'][i]

            return jnp.stack([compute_input(i) for i in range(n_inputs)]).reshape(n_inputs, -1)

        @jit
        def apply_activation(x, activation_id):
            return lax.switch(activation_id, activations, x)

        @jit
        def process_hidden_node(weights, x, bias, slope, slope_mod, activation_id, active_flag):
            net = jnp.dot(weights, x)
            return apply_activation((net + bias) * (slope + slope_mod), activation_id) * active_flag

        @jit
        def process_output_node(weights, x, bias, slope, slope_mod):
            net = jnp.dot(weights, x)
            return (net + bias) * (slope + slope_mod)

        @jit
        def hsl_to_rgb(h, s, l):
            def f(n):
                k = (n + h * 6.0) % 6.0
                a = s * jnp.minimum(l, 1.0 - l)
                return l - a * jnp.maximum(-1.0, jnp.minimum(jnp.minimum(k - 3.0, 9.0 - k), 1.0))

            return jnp.stack([f(0.0), f(2.0), f(4.0)], axis=0)

        @jit
        def run_network(coord_x, coord_y, state, cyclic_start):
            weights = state['weights'] * state['adj_matrix']


            # add weight modulation for cv override
            cv_override = state['cv_override']
            weights_mods = state['weight_mods'] * state['adj_matrix']
            mask = jnp.expand_dims(cv_override, axis=0)  # shape (1, n_nodes)
            weights = weights + weights_mods * mask

            inputs_flat = process_inputs(state, coord_x, coord_y, state['cam_img'])

            x = jnp.zeros((n_nodes, inputs_flat.shape[1]))
            x = x.at[:n_inputs].set(inputs_flat)

            def hidden_cycle(i_cycle, x):
                start = lax.select(i_cycle == 0, n_inputs, cyclic_start)
                end = n_inputs + n_hidden

                def hidden_loop(i, x):
                    hidden_idx = i - n_inputs
                    return x.at[i].set(
                        process_hidden_node(
                            weights[:, i],
                            x,
                            state['biases'][hidden_idx],
                            state['slopes'][hidden_idx],
                            state['slope_mods'][hidden_idx],
                            state['activation_ids'][hidden_idx],
                            state['node_active'][i]
                        )
                    )

                return lax.fori_loop(start, end, hidden_loop, x)

            x = lax.fori_loop(0, self.n_cycles, hidden_cycle, x)

            def output_loop(i, outputs):
                idx = n_inputs + n_hidden + i
                val = process_output_node(
                    weights[:, idx],
                    x,
                    state['output_biases'][i],
                    state['output_slopes'][i],
                    state['output_slope_mods'][i]
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

            H, W = coord_x.shape[0], coord_x.shape[1]
            return out.reshape(3, H, W).transpose(1, 2, 0)

        self.jitted_process_network = run_network

    def sample_network(self):
        while True:
            np_adj_matrix = self.sample_adj_matrix()
            G = nx.DiGraph(np_adj_matrix)

            # Ensure at least one input→output path
            if not any(nx.has_path(G, i, o) for i in self.input_ids[2:] for o in self.output_ids):
                continue

            # Ensure every hidden node lies on some input→hidden→output path
            all_hidden_connected = True
            for h in self.middle_ids:
                if not any(nx.has_path(G, i, h) for i in self.input_ids):
                    all_hidden_connected = False
                    break
                if not any(nx.has_path(G, h, o) for o in self.output_ids):
                    all_hidden_connected = False
                    break

            if all_hidden_connected:
                break

        # Reorder nodes: inputs first, then acyclic hidden nodes, then cyclic hidden nodes, then outputs
        ordered = list(self.input_ids)
        available = set(self.middle_ids)

        while available:
            added = False
            for node in sorted(available):
                sources = np.where(np_adj_matrix[:, node])[0]
                if all(s in ordered for s in sources):
                    ordered.append(node)
                    available.remove(node)
                    added = True
                    break
            if not added:
                break  # cycle detected

        cyclic_start = len(ordered)
        ordered.extend(sorted(available))  # cyclic hidden nodes
        ordered.extend(self.output_ids)

        # Apply node reordering to adj matrix
        reordered_adj = np.zeros_like(np_adj_matrix)
        for i, old_i in enumerate(ordered):
            for j, old_j in enumerate(ordered):
                reordered_adj[i, j] = np_adj_matrix[old_i, old_j]

        # Finalize
        key = random.PRNGKey(int(time.time()))
        self.weights = (random.uniform(key, (self.n_nodes, self.n_nodes)) * 2 - 1) * reordered_adj
        self.activation_ids = jnp.array(np.random.randint(0, len(self.activations), size=self.n_hidden), dtype=jnp.int32)
        self.adj_matrix = jnp.array(reordered_adj)
        self.cyclic_start = cyclic_start
        self.needs_update = True
        print(f"[CPPN] Sampled new network with cyclic start at {self.cyclic_start}")

    def sample_adj_matrix(self):
        adj = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_inputs + self.n_hidden):  # no output→anything
            for j in range(self.n_inputs, self.n_nodes):
                if i == 0 or i == 1:
                    continue
                if i in self.input_ids and j in self.output_ids:
                    continue
                # Allow cycles: no j > i restriction
                prob = 0.4 if j >= self.n_inputs + self.n_hidden else 0.2  # prioritize middle→output
                if np.random.rand() < prob:
                    adj[i, j] = 1
        return adj


    def save_state(self):
        timestamp = self.timestamp
        pkl_path = f"{self.output_path}/state_{timestamp}.pkl"
        img_path = f"{self.output_path}/image_{timestamp}.png"
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.state, f)
        img = self.update(res=2048)
        display_image = np.array((img * 255).astype(jnp.uint8))
        print(f'Checkpoints saved at {img_path}')
        im = Image.fromarray(display_image)
        im.save(img_path)
        return img

    @property
    def timestamp(self):
        return time.strftime("%Y_%m_%d_%H%M%S")

    @property
    def state(self):
        keys = ['adj_matrix', 'biases', 'slopes', 'slope_mods', 'weights', 'activation_ids', 'output_slopes', 'output_slope_mods', 'output_biases', 'output_modes',
                'input_function_ids', 'input_params1', 'input_params2', 'node_active', 'cv_override',
                'inputs_nodes_record', 'cyclic_start']
        state = {}
        for key in keys:
            state[key] = self.__dict__[key]
        return state

    def set_state(self, state=None, state_path=None):
        assert state != None or state_path != None
        if state is None:
            with open(state_path, "rb") as f:
                state = pickle.load(f)
        for key, val in state.items():
            self.__dict__[key] = val

    @property
    def time(self):
        return time.time() - self.t_start


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