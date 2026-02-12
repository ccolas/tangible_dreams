import pickle
import jax.numpy as jnp
from jax import random
from copy import deepcopy
from collections import deque
import networkx as nx
import numpy as np
import time
from jax import jit
import jax
from functools import partial
from PIL import Image
from src.save.audio_proc import AudioProcessor
print(jax.devices())  # lists all available devices (CPU, GPU, TPU)
print(jax.default_backend())  # shows the default backend being used

class CPPN:
    def __init__(self, output_path, params, maxlen=None):
        self.output_path = output_path
        self.params = params
        self.n_inputs = 3
        self.n_hidden = 8
        self.n_outputs = 3
        self.major_history = deque(maxlen=maxlen)  # For network resamples
        self.minor_history = deque(maxlen=100)  # For M/R/S button changes
        self.major_index = -1
        self.minor_index = -1
        self.inputs = dict()
        self.needs_update = True
        self.first = True

        # Initialize node biases
        self.biases = jnp.zeros(self.n_hidden)
        # Initialize node multipliers (for knobs)
        self.multipliers = jnp.ones(self.n_hidden)
        # Add new parameters
        self.x_offset = 0
        self.y_offset = 0
        self.zoom = 1.0
        self.rgb_slopes = jnp.ones(3)  # For RGB output scaling
        self.rgb_biases = jnp.zeros(3)  # For RGB sigmoid centering
        # Add SET button state
        self.set_pressed = False
        self.res = params['res']
        self.factor = params['factor']
        for res in [32, 64, 128, 256, 512, 1024, 2048]:
            self.inputs[res] = self.generate_inputs(res)


        self.n_nodes = self.n_inputs + self.n_hidden + self.n_outputs
        self.adj_matrix = np.zeros((self.n_nodes, self.n_nodes))
        # Add audio processor
        self.audio = AudioProcessor(n_bands=8)
        self.audio.start()

        # Add audio reactive parameters
        self.audio_multipliers = jnp.ones(self.n_hidden)
        self.audio_biases = jnp.zeros(self.n_hidden)
        self.audio_rgb = jnp.ones(3)

        @jit
        def weierstrass(x):
            # Parameters calibrated for interesting output
            a = 0.5  # Amplitude decay (0.4-0.6 works well)
            b = 3.0  # Frequency multiplier (2-4 works well)

            # Sum multiple cosine waves with increasing frequency and decreasing amplitude
            result = jnp.zeros_like(x)
            for i in range(4):
                # Each iteration adds a higher frequency component
                # Add phase shift to create more variation
                result = result + a ** i * jnp.cos(b ** i * x + jnp.pi / 3)

            # Scale output to reasonable range
            return result * 0.4  # Scale factor to keep output roughly in [-1,1]

        @jit
        def takagi(x):
            # Scale input for good variation
            x = x * 0.5

            # Parameters
            w = 0.6  # Weight decay factor
            n = 4  # Number of iterations

            # Prepare weights and reshape for broadcasting
            i_vals = jnp.arange(n, dtype=jnp.float32).reshape(-1, 1)  # Make column vector
            weights = w ** i_vals
            scales = 2.0 ** i_vals

            # Broadcasting will now work correctly with x
            scaled_x = scales * x
            terms = weights * jnp.abs(scaled_x - jnp.round(scaled_x))

            # Sum along axis 0 (the i_vals dimension)
            result = jnp.sum(terms, axis=0)

            return result - 0.5  # Center the output range

        @jit
        def riemann(x):
            # Scale input for good coverage of sin function
            x = x * 3.0

            # Create the i values and reshape for broadcasting
            i_vals = jnp.arange(1.0, 5.0)  # [1,2,3,4]
            i_vals = i_vals.reshape(-1, 1)  # Make it a column vector for broadcasting

            # Now when we multiply with x, broadcasting will work correctly
            # x will be treated as a row vector
            squared_i = i_vals * i_vals
            terms = (1.0 / squared_i) * jnp.sin(squared_i * x)

            # Sum along the first axis (the i values)
            result = jnp.sum(terms, axis=0) * 0.7

            return result

        self.activations = [
            # lambda x: jnp.tanh(x * 3),  # 0: tanh with good range
            lambda x: jnp.sin(x) + jnp.sin(2.4 * x),
            lambda x: jnp.tanh(x * 3),  # 0: tanh with good range
            # lambda x: jnp.sin(x) + jnp.sin(2.4 * x),
            # lambda x: x,
                # lambda x: jnp.clip(x, -1, 1),  # 1: linear bounded
                # lambda x: jnp.abs(x * 0.7),  # 2: abs with good scaling
            lambda x: jnp.cos(x * 2.3),  # 3: cos with good oscillation
                # lambda x: jax.lax.select(  # 4: sinxx with handling for x=0
                #     jnp.abs(x) < 1e-4,
                #     jnp.ones_like(x),
                #     jnp.sin(x * 6.67) / (x * 6.67)
                # ),
            # lambda x: jnp.sin(x) + 0.5 * jnp.sin(2 * x) + 0.25 * jnp.sin(4 * x),  # 5: perlin-like
                # lambda x: x * (1 / (1 + jnp.exp(-x * 1.5))),  # 6: swish
            # lambda x: jnp.clip(((x / 5) % 0.5) / 0.5, -1, 1),  # 7: modulo
            # lambda x: jnp.exp(-(x * x) / 1.0),  # 8: gaussian
            # # Weierstrass function (a=0.7, b=4, n=4)
            # lambda x: weierstrass(x * 2.0),  # Scale input for good variation
            #
            # # Riemann function (n=4)
            # lambda x: riemann(x),
            # lambda x: takagi(x),
        ]



        self.sample_network()  # Initialize weights and connections
        # Initialize JIT-compiled functions
        self._init_jitted_functions()


    def _init_jitted_functions(self):
        # Compile core computations

        @jit
        def apply_activation(x, activation_id):
            """Apply activation function based on ID using JAX's switch"""
            return jax.lax.switch(activation_id, self.activations, x)

        @jit
        def process_hidden_node(weights, x, multiplier, bias, activation_id):
            net = jnp.dot(weights, x) * multiplier + bias
            return apply_activation(net, activation_id)

        @jit
        def process_output_node(weights, x, slope, bias):
            net = (jnp.dot(weights, x) + bias) * slope
            return 1.0 / (1.0 + jnp.exp(-net))

        @partial(jit, static_argnums=(3, 4, 5))
        def process_network(x, weights, state_dict, n_inputs, n_hidden, cyclic_start):
            multipliers = state_dict['multipliers']
            biases = state_dict['biases']
            rgb_slopes = state_dict['rgb_slopes']
            rgb_biases = state_dict['rgb_biases']
            activation_ids = state_dict['activation_ids']  # Add this

            def body_fun(i_cycle, x):
                start = jnp.where(i_cycle == 0, n_inputs, cyclic_start)

                # Process hidden nodes
                def hidden_loop(i, x):
                    node_idx = i - n_inputs
                    x = x.at[i].set(
                        process_hidden_node(
                            weights[:, i],
                            x,
                            multipliers[node_idx],
                            biases[node_idx],
                            activation_ids[node_idx]  # Pass activation ID
                        )
                    )
                    return x

                x = jax.lax.fori_loop(start, n_inputs + n_hidden, hidden_loop, x)

                # Process output nodes
                def output_loop(i, x):
                    out_idx = i - (n_inputs + n_hidden)
                    x = x.at[i].set(
                        process_output_node(
                            weights[:, i],
                            x,
                            rgb_slopes[out_idx],
                            rgb_biases[out_idx]
                        )
                    )
                    return x

                x = jax.lax.fori_loop(
                    n_inputs + n_hidden,
                    weights.shape[0],
                    output_loop,
                    x
                )
                return x

            x = jax.lax.fori_loop(0, 2, lambda i, x: body_fun(i, x), x)
            return x

        self.jitted_process_network = process_network

    def define_coordinate_inputs(self, x, y):
        """Dictionary of all possible coordinate transforms"""
        import jax.numpy as jnp

        # Basic coordinates
        basic = {
            'x': x,
            'y': y,
            "x_sim": jnp.abs(x - x.mean()),
            "y_sim": jnp.abs(y - y.mean()),
            'dist': jnp.sqrt(x ** 2 + y ** 2) / np.sqrt(2),  # Euclidean distance from center
        }

        # Geometric/Spatial
        geometric = {
            'angle': jnp.arctan2(y, x) / jnp.pi,  # Normalized to [-1,1]
            'manhattan_dist': jnp.abs(x) + jnp.abs(y),
            'x_squared': x ** 2 * jnp.sign(x),  # Maintain sign for smoothness
            'y_squared': y ** 2 * jnp.sign(y),
            'x_cubed': x ** 3,
            'y_cubed': y ** 3,
            'radial_wave_1': jnp.sin(2 * jnp.sqrt(x ** 2 + y ** 2)),
            'radial_wave_2': jnp.sin(4 * jnp.sqrt(x ** 2 + y ** 2)),
            'grid_1': jnp.sin(x * 3.14159) * jnp.sin(y * 3.14159),
            'grid_2': jnp.sin(x * 6.28318) * jnp.sin(y * 6.28318),
        }

        # Symmetry-based
        symmetry = {
            'abs_x': jnp.abs(x),
            'abs_y': jnp.abs(y),
            'min_coord': jnp.minimum(jnp.abs(x), jnp.abs(y)),
            'max_coord': jnp.maximum(jnp.abs(x), jnp.abs(y)),
            'dist_from_x_axis': jnp.abs(y),
            'dist_from_y_axis': jnp.abs(x),
            'dist_from_diagonal': jnp.abs(x - y) / jnp.sqrt(2),
        }

        # Polar-based
        r = jnp.sqrt(x ** 2 + y ** 2)
        theta = jnp.arctan2(y, x)
        polar = {
            'r': r,
            'theta': theta / jnp.pi,  # Normalized to [-1,1]
            'r_mod_1': jnp.mod(r * 2, 1.0) * 2 - 1,  # Normalized to [-1,1]
            'r_mod_2': jnp.mod(r * 4, 1.0) * 2 - 1,
            'spiral_1': jnp.sin(theta + 5 * r),
            'spiral_2': jnp.sin(2 * theta + 3 * r),
        }

        # Compound patterns
        compound = {
            'ripple': jnp.sin(10 * (x ** 2 + y ** 2)),
            'hyperbolic': x * y / (jnp.sqrt(x ** 2 + y ** 2) + 0.1),
            'gaussian': jnp.exp(-(x ** 2 + y ** 2)),
            'diamond': jnp.maximum(jnp.abs(x) + jnp.abs(y), jnp.abs(x - y)),
            'star': jnp.sin(jnp.arctan2(y, x) * 5),
        }

        # Combine all dictionaries
        all_inputs = {}
        all_inputs.update(basic)
        all_inputs.update(geometric)
        all_inputs.update(symmetry)
        all_inputs.update(polar)
        all_inputs.update(compound)

        return all_inputs

    def generate_inputs(self, res):
        x = jnp.linspace(-1, 1, int(res * self.factor))
        y = jnp.linspace(-1, 1, res)
        X, Y = jnp.meshgrid(x, y)

        # Apply transformations
        X = (X / self.zoom) + self.x_offset
        Y = (Y / self.zoom) + self.y_offset

        # Get all possible inputs
        input_maps = self.define_coordinate_inputs(X, Y)

        # Choose which inputs to use
        selected_inputs = [
            # input_maps['x'],
            # input_maps['y'],
            input_maps['abs_x'],
            input_maps['abs_y'],
            input_maps['dist']
            # input_maps['abs_x']
        ]
        self.n_inputs = len(selected_inputs)
        output = jnp.stack(selected_inputs).reshape(self.n_inputs, -1)

        # if self.with_cam:
        #     # Get camera input
        #     camera = self.get_camera_input(res, int(res * self.factor))
        #     camera = jnp.array(camera)  # Convert to JAX array
        #     output = jnp.stack([X, Y, dist, camera]).reshape(self.n_inputs, -1)
        # else:
        #     output = jnp.stack([X, Y, dist]).reshape(self.n_inputs, -1)
        return output



    def sample_adj_matrix(self):
        """Sample and reorder adjacency matrix for optimal computation order"""

        # First sample adjacency matrix
        self.adj_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_inputs + self.n_hidden):
            for j in range(self.n_inputs, self.n_nodes):
                if j > i:
                    prob = 0.5
                else:
                    prob = 0.2
                if np.random.rand() < prob:
                    self.adj_matrix[i, j] = 1

        # Compute optimal ordering for hidden nodes
        ordered_nodes = list(range(self.n_inputs))  # Start with inputs
        available_nodes = set(range(self.n_inputs, self.n_inputs + self.n_hidden))
        while available_nodes:
            # Find nodes that only depend on what's already ordered
            found_one = False
            for node in available_nodes:
                input_nodes = set(np.where(self.adj_matrix[:, node])[0])
                if input_nodes.issubset(set(ordered_nodes)):
                    ordered_nodes.append(node)
                    available_nodes.remove(node)
                    found_one = True
                    break

            if not found_one:
                # We hit the cyclic group
                self.cyclic_start = len(ordered_nodes)
                # Add remaining nodes in current order
                ordered_nodes.extend(sorted(available_nodes))
                break

        # Add output nodes at the end
        ordered_nodes.extend(range(self.n_inputs + self.n_hidden, self.n_nodes))

        # Reorder adjacency matrix according to new ordering
        self.node_order = ordered_nodes
        reordered_adj = np.zeros_like(self.adj_matrix)
        for i, old_i in enumerate(ordered_nodes):
            for j, old_j in enumerate(ordered_nodes):
                reordered_adj[i, j] = self.adj_matrix[old_i, old_j]

        self.adj_matrix = reordered_adj
        print(f"Cyclic dependencies start at node {self.cyclic_start}")

    def sample_network(self):
        key = random.PRNGKey(int(time.time()))

        path_found = False
        while not path_found:
            self.sample_adj_matrix()
            # Ensure path from inputs to outputs
            G = nx.Graph(self.adj_matrix)
            for inp in range(self.n_inputs):
                for out in range(self.n_inputs + self.n_hidden, self.n_nodes):
                    if nx.has_path(G, inp, out):
                        path_found = True
                        break
                if path_found:
                    break

        # For weights, use uniform sampling between -1 and 1
        self.weights = (random.uniform(key, (self.n_nodes, self.n_nodes)) * 2 - 1) * self.adj_matrix

        # For activation functions, randomly assign one to each hidden node at start
        self.activation_ids = random.randint(key, (self.n_hidden,), 0, len(self.activations)).astype(int)

        # Reset node parameters
        self.biases = jnp.zeros(self.n_hidden)
        self.multipliers = jnp.ones(self.n_hidden)
        # Add new parameters
        self.x_offset = 0
        self.y_offset = 0
        self.zoom = 1.0
        self.rgb_slopes = jnp.ones(3)  # For RGB output scaling
        self.rgb_biases = jnp.zeros(3)  # For RGB sigmoid centering

        # Save to major history
        self.save_major_state()
        self.needs_update = True
        print(f'Sampling new network idx {self.major_index}')

    def add_connection(self, node_idx):
        """Add a random outgoing connection from the specified hidden node"""
        actual_idx = self.n_inputs + node_idx
        # Only consider nodes after this one (forward connections)
        possible_targets = list(range(actual_idx + 1, self.n_nodes))
        # Filter out already connected nodes
        possible_targets = [t for t in possible_targets if self.adj_matrix[actual_idx, t] == 0]

        if possible_targets:
            target = np.random.choice(possible_targets)
            self.adj_matrix[actual_idx, target] = 1
            # Initialize with random weight
            self.weights = self.weights.at[actual_idx, target].set(np.random.normal(0, 1))
            self.save_minor_state()
            self.needs_update = True
            print(f'Add connection {node_idx}')

    def remove_connection(self, node_idx):
        """Remove a random outgoing connection from the specified hidden node"""
        actual_idx = self.n_inputs + node_idx
        # Get existing outgoing connections
        targets = np.where(self.adj_matrix[actual_idx, :] == 1)[0]

        if len(targets) > 0:
            # Check if removing connection would break path to output
            temp_matrix = self.adj_matrix.copy()
            valid_targets = []

            for target in targets:
                temp_matrix[actual_idx, target] = 0
                G = nx.Graph(temp_matrix)
                path_exists = False
                for inp in range(self.n_inputs):
                    for out in range(self.n_inputs + self.n_hidden, self.n_nodes):
                        if nx.has_path(G, inp, out):
                            path_exists = True
                            break
                    if path_exists:
                        break
                if path_exists:
                    valid_targets.append(target)
                temp_matrix[actual_idx, target] = 1

            if valid_targets:
                target = np.random.choice(valid_targets)
                self.adj_matrix[actual_idx, target] = 0
                self.weights = self.weights.at[actual_idx, target].set(0)
                self.save_minor_state()
                self.needs_update = True
                print(f'Remove connection {node_idx}')

    def resample_out_connections(self, node_idx):
        """Resample all outgoing connections from the specified hidden node"""
        actual_idx = self.n_inputs + node_idx

        # Clear existing outgoing connections
        self.adj_matrix[actual_idx, :] = 0
        self.weights = self.weights.at[actual_idx, :].set(0)

        # Randomly add new connections (similar to sample_adj_matrix logic)
        for j in range(actual_idx + 1, self.n_nodes):  # Only forward connections
            if np.random.rand() < 0.5:  # 50% chance for each possible connection
                self.adj_matrix[actual_idx, j] = 1
                self.weights = self.weights.at[actual_idx, j].set(np.random.uniform(-1, 1))

        # Check if any path to output exists
        G = nx.Graph(self.adj_matrix)
        path_exists = False
        for inp in range(self.n_inputs):
            for out in range(self.n_inputs + self.n_hidden, self.n_nodes):
                if nx.has_path(G, inp, out):
                    path_exists = True
                    break
            if path_exists:
                break

        # If no path exists, try again
        if not path_exists:
            self.resample_out_connections(node_idx)
        else:
            self.save_minor_state()
            self.needs_update = True

    def resample_in_connections(self, node_idx):
        """Resample all incoming connections to the specified hidden node"""
        actual_idx = self.n_inputs + node_idx

        # Clear existing incoming connections
        self.adj_matrix[:actual_idx, actual_idx] = 0
        self.weights = self.weights.at[:actual_idx, actual_idx].set(0)

        # Add new random connections from previous layers
        for i in range(actual_idx):  # Only from previous nodes
            if np.random.rand() < 0.5:  # 50% chance for each possible connection
                self.adj_matrix[i, actual_idx] = 1
                self.weights = self.weights.at[i, actual_idx].set(np.random.uniform(-1, 1))

        self.save_minor_state()
        self.needs_update = True

    def update(self, res=None):
        t_init = time.time()
        t_init_total = time.time()
        times = dict()


        # Collect state into dict for JIT
        state_dict = {
            'multipliers': self.multipliers ,
            'biases': self.biases,
            'rgb_slopes': self.rgb_slopes,
            'rgb_biases': self.rgb_biases,
            'activation_ids': self.activation_ids,
        }


        if res is not None:
            inputs = self.generate_inputs(res)
        else:
            res = self.res
            inputs = self.inputs[res]

        # Pre-allocate activation array
        x = jnp.zeros((self.n_nodes, inputs.shape[1]))
        x = x.at[:self.n_inputs].set(inputs)


        times['pre'] = time.time() - t_init
        t_init = time.time()
        # Process network using JIT-compiled function
        x = self.jitted_process_network(
            x,
            self.weights,
            state_dict,
            self.n_inputs,
            self.n_hidden,
            self.cyclic_start
        )
        times['main'] = time.time() - t_init
        t_init = time.time()
        # Extract and reshape output
        rgb = x[-self.n_outputs:]
        h, w = res, int(res * self.factor)
        self.current_image = rgb.reshape(3, h, w).transpose(1, 2, 0)
        times['post']= time.time() - t_init
        times['total'] = time.time() - t_init_total
        print('times')
        for k, v in times.items():
            print('  ', k, v)
        if not self.first:
            self.needs_update = False
        self.first = False
        return self.current_image

    def save_major_state(self):
        # Clear future states if we're not at the end
        while len(self.major_history) > self.major_index + 1:
            self.major_history.pop()
        self.major_history.append(deepcopy(self.state))
        self.major_index = len(self.major_history) - 1
        # Reset minor history
        self.minor_history.clear()
        self.minor_index = -1

    def save_minor_state(self):
        # Clear future states if we're not at the end
        while len(self.minor_history) > self.minor_index + 1:
            self.minor_history.pop()
        self.minor_history.append(deepcopy(self.state))
        self.minor_index = len(self.minor_history) - 1

    @property
    def state(self):
        state = {
            'weights': self.weights,
            'adj_matrix': self.adj_matrix,
            'activation_ids': self.activation_ids,
            'biases': self.biases,
            'multipliers': self.multipliers,
            'x_offset': self.x_offset,
            'y_offset': self.y_offset,
            'zoom': self.zoom,
            'rgb_slopes': self.rgb_slopes,
            'rgb_biases': self.rgb_biases,
            'cyclic_start': self.cyclic_start,  # Add this
            'node_order': self.node_order  # Add this
        }
        return state

    @property
    def timestamp(self):
        return time.strftime("%Y_%m_%d_%H%M%S")

    def set_state(self, state=None, state_path=None):
        assert state != None or state_path != None
        if state is None:
            with open(state_path, "rb") as f:
                state = pickle.load(f)
        for key, val in state.items():
            self.__dict__[key] = val

    def save_state(self):
        
        timestamp = self.timestamp
        pkl_path = f"{self.output_path}/state_{timestamp}.pkl"
        img_path = f"{self.output_path}/image_{timestamp}.png"
        print(pkl_path)
        print(img_path)
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.state, f)
        img = self.update(res=2048)
        print(f'Saving output to {img_path}')
        # Convert to uint8 range [0, 255]
        np_img = np.array(img)
        img_uint8 = (np_img * 255).astype('uint8')
        # Save using PIL
        im = Image.fromarray(img_uint8)
        im.save(img_path)

    def load_major_state(self):
        if 0 <= self.major_index < len(self.major_history):
            state = self.major_history[self.major_index]
            self.set_state(state)
            # Reset minor history when loading major state
            self.minor_history.clear()
            self.minor_index = -1
            self.needs_update = True

    def load_minor_state(self):
        if 0 <= self.minor_index < len(self.minor_history):
            state = self.minor_history[self.minor_index]
            self.set_state(state)
            self.needs_update = True

    # def __del__(self):
    #     if self.has_camera:
    #         self.cap.release()

    def __del__(self):
        # Clean up audio processor
        if hasattr(self, 'audio'):
            self.audio.stop()