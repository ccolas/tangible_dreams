import time
import numpy as np
import jax.numpy as jnp
from jax import random
from PIL import Image
import pickle
import jax

from viz import create_backend
from src.cppn_utils import activation_fns, build_coordinate_grid, input_functions, input_selector_mapping, zoom_mapping, bias_mapping, noise
from save.misc.error_screen import generate_error_screen
from save.misc.compute_n_cycles import compute_rounds
from src.sound_input import AudioReactive


SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
BANDS = [(20,200), (200,2000), (2000,8000)]  # bass, mid, high

def _band_energy(mag, freqs, band):
    lo, hi = band
    idx = np.where((freqs >= lo) & (freqs < hi))[0]
    return np.mean(mag[idx]) if len(idx) else 0.0


class CPPN:
    def __init__(self, output_path, params):
        self.output_path = output_path
        self.params = params
        self.t_start = time.time()
        self.update_period = 0.01
        self.cam_input = False

        # General config
        self.default_res = 1024
        self.high_res = 1900
        self.factor = params['factor']
        self.debug = params.get('debug', False)

        self.reactivity = params['reactivity']
        # Architecture
        self.n_inputs = 5
        self.n_hidden = 9
        self.n_outputs = 3
        self.x_id = 4
        self.y_id = 3
        self.n_nodes = self.n_inputs + self.n_hidden + self.n_outputs
        self.n_cycles = 1

        self.input_ids = list(range(0, self.n_inputs))
        self.middle_ids = list(range(self.n_inputs, self.n_inputs + self.n_hidden))
        self.output_ids = list(range(self.n_inputs + self.n_hidden, self.n_nodes))

        self.last_times = [0 for _ in range(self.n_nodes)]
        self.audio_values = np.zeros(len(BANDS))  # [bass, mid, high]
        self.inputs_nodes_record = np.full((self.n_nodes, 3), -1, dtype=int)  # track what gets connected on inputs

        # Activations
        self.activations = list(activation_fns.values())
        self.activations_names = list(activation_fns.keys())

        # Precomputed coordinate grids
        self.input_function_names = input_selector_mapping
        self.input_functions = [input_functions[name] for name in self.input_function_names]
        self.coords = {self.default_res: build_coordinate_grid(self.default_res, self.factor),
                       self.high_res: build_coordinate_grid(self.high_res, self.factor)}
        self.error_screen = {self.default_res: generate_error_screen(self.default_res, self.factor),
                             self.high_res: generate_error_screen(self.high_res, self.factor)}

        self.device_state = self.make_initial_state()
        # Camera input
        # self.cam = cv2.VideoCapture(0)  # webcam index 0
        # self.cam_img = np.zeros(self.img_shapes[self.default_res], dtype=np.float32)  # normalized grayscale

        # Runtime
        self.needs_update = True

        # Placeholder for compiled forward pass
        self.jitted_process_network = None
        self._init_jitted_functions()
        self.is_valid = False
        self.last_update_saved = False
        self.reactive_states = {i: {} for i in range(self.n_nodes)}
        self._x_in = None  # cached (N, n_inputs) on device
        self._x_in_key = None  # small host tuple to know when to invalidate
        self.weights_1_raw = np.zeros([self.n_nodes])  # raw value of weight input 1 (weight 3?)



        if params.get('load_from'):
            self.set_state(state_path=params['load_from'])

        if "audio" in self.reactivity:
            self.audio = AudioReactive(visualize=params['visualize_audio'])
            self.audio.start()
        else:
            self.audio = None

    def make_initial_state(self):
        state = {
            'weights': jnp.zeros((self.n_nodes, self.n_nodes), dtype=jnp.float16),
            'weight_mods': jnp.zeros((self.n_nodes, self.n_nodes), dtype=jnp.float16),
            'adj_matrix': jnp.zeros((self.n_nodes, self.n_nodes), dtype=jnp.bool_),
            'input_function_ids': jnp.array([-1] * self.n_inputs, dtype=jnp.int32),
            'inverted_inputs': jnp.zeros(self.n_inputs, dtype=jnp.int32),
            'input_params1': jnp.full(self.n_inputs, 512, dtype=jnp.int32),
            'input_params2': jnp.full(self.n_inputs, 512, dtype=jnp.int32),
            'node_active': jnp.ones(self.n_nodes, dtype=jnp.bool_),
            'biases': jnp.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ], jnp.float32), #jnp.zeros(self.n_hidden, dtype=jnp.float32),
            'slopes': jnp.ones(self.n_hidden, dtype=jnp.float32),
            'slope_mods': jnp.zeros(self.n_hidden, dtype=jnp.float32),
            'activation_ids': jnp.zeros(self.n_hidden, dtype=jnp.int32),
            'output_biases': jnp.zeros(self.n_outputs, dtype=jnp.float32),
            'output_slopes': jnp.ones(self.n_outputs, dtype=jnp.float32),
            'output_slope_mods': jnp.zeros(self.n_outputs, dtype=jnp.float32),
            'cv_override': jnp.zeros(self.n_nodes, dtype=jnp.bool_),
        }
        state['input_function_ids'] = state['input_function_ids'].at[self.x_id].set(0)
        state['input_function_ids'] = state['input_function_ids'].at[self.y_id].set(1)
        return state

    def reactive_update(self, i, w):
        """
        Reactive function per node.
        - i: node index (0-based)
        - w: normalized control in [0,1] from weight 1 (sensor value / 1023)
        """
        w /= 1023
        now = self.time
        if now - self.last_times[i] <= self.update_period:
            return None
        self.last_times[int(i)] = now

        period = 4.0
        omega = 2 * np.pi / period / 2
        A = 2.0 * w

        # === Outputs: all same simple sine ===
        if i in self.output_ids:
            return A * np.sin(omega * now + i)

        # === Middles: 9 different dynamics ===
        idx = i - self.n_inputs  # 0..8
        state = self.reactive_states[i]


        # === Time-varying nodes: idx 0, 3, 6 (global 5, 8, 11) ===
        if idx == 0 and "time" in self.reactivity:  # sine + harmonics
            return w * (np.sin(omega * now) + 0.15 * np.sin(omega * 7 * now))
        elif idx == 3 and "time" in self.reactivity:  # Perlin-modulated sine
            drift = 0.3 * noise.noise1(now * 0.05, octaves=2)
            return A * np.sin((omega * (1 + drift)) * now)
        elif idx == 6 and "time" in self.reactivity:  # Perlin drift
            t = now * 0.2 + idx * 10.0
            return A * (2 * noise.noise1(t, octaves=3, persistence=0.2, lacunarity=2.0) - 1.0)

        # === Audio nodes (6 signals from EMA + spectral flux) ===
        # audio_array: [bass_ema, bass_flux, mid_ema, mid_flux, treble_ema, treble_flux]
        #   idx 1 (node 6):  bass_ema   → arr[0]
        #   idx 2 (node 7):  bass_flux  → arr[1]
        #   idx 4 (node 9):  mid_ema    → arr[2]
        #   idx 5 (node 10): mid_flux   → arr[3]
        #   idx 7 (node 12): treble_ema → arr[4]
        #   idx 8 (node 13): treble_flux→ arr[5]
        elif idx in (1, 2, 4, 5, 7, 8) and "audio" in self.reactivity:
            audio_idx = {1: 0, 2: 1, 4: 2, 5: 3, 7: 4, 8: 5}[idx]
            audio_val = float(self.audio.audio_array[audio_idx])
            return A * (audio_val * 2 - 1)

        # Fallback: time-varying Perlin for any unmatched node
        elif "time" in self.reactivity:
            t = now * 0.2 + idx * 10.0
            return A * (2 * noise.noise1(t, octaves=3, persistence=0.2, lacunarity=2.0) - 1.0)

        return 0.0

    def update(self, res=None):
        self.last_update_saved = False
        if res is None:
            res = self.default_res
        coords_x, coords_y = self.coords[res]

        key = self._inputs_key(res)
        if (self._x_in is None) or (key != self._x_in_key):
            print(f"  [CPPN] recomputing inputs (cache miss)")
            self._x_in = self._compute_inputs(self.device_state, coords_x, coords_y)  # (N, n_in)
            self._x_in_key = key

        img = self._run_cycles(self.device_state, self._x_in, coords_x, self.n_cycles)  # uint8 on device
        self.needs_update = False
        return img

    def _init_jitted_functions(self):
        n_in, n_h, n_out = self.n_inputs, self.n_hidden, self.n_outputs
        n_nodes = n_in + n_h + n_out

        activations = tuple(self.activations)  # static inside JIT
        input_fns = tuple(self.input_functions)  # static inside JIT

        from jax import jit, lax
        import jax
        import jax.numpy as jnp


        @jit
        def _basis_switch(fn_id, x, y, p1, p2):
            return lax.switch(fn_id, input_fns, x, y, p1, p2)

        @jit
        def _compute_inputs(state, coords_x, coords_y):
            # Build (N, n_in) once per input-param change
            gz_x = zoom_mapping(state['input_params1'][self.x_id])
            gz_y = zoom_mapping(state['input_params1'][self.y_id])
            gb_x = bias_mapping(state['input_params2'][self.x_id])
            gb_y = bias_mapping(state['input_params2'][self.y_id])

            x = coords_x
            y = coords_y
            H, W = x.shape
            N = H * W

            def one_input(i):
                p1 = state['input_params1'][i]
                p2 = state['input_params2'][i]
                f_id = state['input_function_ids'][i]
                inv = (state['inverted_inputs'][i] > 0)

                x_i = (x + gb_x) / gz_x
                y_i = (y + gb_y) / gz_y
                x_i = lax.select(inv & (f_id == 0), -x_i, x_i)
                y_i = lax.select(inv & (f_id == 1), -y_i, y_i)

                return _basis_switch(f_id, x_i, y_i, p1, p2) * state['node_active'][i]

            inp = jnp.stack([one_input(i) for i in range(n_in)], axis=0)  # (n_in,H,W)
            return inp.reshape(n_in, N).T  # (N, n_in) float32

        @jit
        def _apply_activation_rows(vals, act_ids):
            # vals: (n_h, N) -> apply per hidden row
            def f(aid, row):
                return lax.switch(aid, activations, row)

            return jax.vmap(f, in_axes=(0, 0))(act_ids, vals)

        @jit
        def _run_cycles(state, x_in, coords_x, n_cycles):
            # x_in: (N, n_in); produce uint8 image
            H, W = coords_x.shape
            N = H * W

            active = state['node_active'][:, None]  # (n_nodes,1)

            # weights + mask (+ mods if cv_override enabled)
            Wts = (state['weights'] + state['weight_mods'] * state['cv_override'][None, :])
            Wts = (Wts * state['adj_matrix']).astype(jnp.float16)  # fp16 buffers; accumulate fp32

            # activations buffer: (n_nodes, N) bf16/fp16
            x = jnp.zeros((n_nodes, N), dtype=jnp.float16)
            x = x.at[:n_in].set((x_in.T.astype(jnp.float16)) * active[:n_in])


            def body(i, carry):
                x, _net_prev = carry
                net = jnp.matmul(Wts.T.astype(jnp.float32), x.astype(jnp.float32))  # (n_nodes,N)

                h = net[n_in:n_in + n_h]
                h = (h + state['biases'][:, None]) * (state['slopes'][:, None] + state['slope_mods'][:, None])
                h = _apply_activation_rows(h, state['activation_ids'])
                x = x.at[n_in:n_in + n_h].set((h * active[n_in:n_in + n_h]).astype(jnp.float16))

                return (x, net)

            net0 = jnp.zeros((n_nodes, N), dtype=jnp.float32)
            x, net = lax.fori_loop(0, n_cycles, body, (x, net0))

            out = net[n_in + n_h:n_in + n_h + n_out]
            out = (out + state['output_biases'][:, None]) * (state['output_slopes'][:, None] + state['output_slope_mods'][:, None])
            # out = jax.nn.sigmoid(5 * out)
            out = out * active[n_in + n_h:n_in + n_h + n_out]
            out = jnp.clip(out, 0.0, 2.0)  # (n_out, N)
            img = (out.T * 255.0).astype(jnp.uint8).reshape(H, W, n_out)
            return img

        self._compute_inputs = _compute_inputs
        self._run_cycles = _run_cycles
        self._x_in = None
        self._x_in_key = None

    def update_cycles_if_needed(self):
        """Call when topology changes"""
        adj = self.device_state['adj_matrix'] & self.device_state['node_active'][:, None] & self.device_state['node_active'][None, :]
        self.n_cycles, _ = compute_rounds(adj, self.n_inputs, self.n_outputs, settling_count=1)
        if self.n_cycles is None: self.n_cycles = 1
        print(f'Setting n_cycles to {self.n_cycles}')

    def apply_postprocessing(self, img):
        """Apply post-processing matching the GLSL shader order:
        symmetry → displacement → (sample) → invert → grain
        """
        H, W = img.shape[:2]
        out = img.astype(np.float32) / 255.0

        # Build normalized UV coords [0,1]
        uu, vv = np.meshgrid(np.linspace(0, 1, W, endpoint=False),
                             np.linspace(0, 1, H, endpoint=False))
        uv_x = uu
        uv_y = vv

        # --- symmetry (operates on UV) ---
        if self.symmetry_mode != 0:
            sector_map = {1: 6, 2: 8, 3: 12, 4: 4}
            sectors = sector_map.get(self.symmetry_mode, self.symmetry_mode)
            uv_x, uv_y = self._kaleidoscope_uv(uv_x, uv_y, sectors)

        # --- displacement (operates on UV) ---
        if self.displace_strength > 0.001:
            d = self.displace_strength  # fraction of screen (0-0.05)
            dx = (self._hash2d(uv_x * 200.0, uv_y * 200.0) - 0.5) * d
            dy = (self._hash2d(uv_x * 300.0, uv_y * 300.0) - 0.5) * d
            uv_x = uv_x + dx
            uv_y = uv_y + dy

        # Clamp and sample
        uv_x = np.clip(uv_x, 0.0, 1.0 - 1e-6)
        uv_y = np.clip(uv_y, 0.0, 1.0 - 1e-6)
        ix = (uv_x * W).astype(int)
        iy = (uv_y * H).astype(int)
        out = out[iy, ix]

        # --- invert ---
        if self.invert:
            out = 1.0 - out

        # --- grain (uniform monochrome, matching shader) ---
        if self.grain_strength > 0.001:
            g = (self._hash2d(uv_x * 500.0, uv_y * 500.0) - 0.5) * self.grain_strength
            out = out + g[:, :, np.newaxis]

        out = np.clip(out, 0.0, 1.0)
        return (out * 255).astype(np.uint8)

    @staticmethod
    def _hash2d(x, y):
        """Deterministic hash matching GLSL: fract(sin(dot(co, vec2(12.9898,78.233))) * 43758.5453)"""
        dot = x * 12.9898 + y * 78.233
        return np.mod(np.sin(dot) * 43758.5453, 1.0)

    @staticmethod
    def _kaleidoscope_uv(uv_x, uv_y, sectors):
        """Kaleidoscope matching the GLSL shader."""
        cx, cy = 0.5, 0.5
        px = uv_x - cx
        py = uv_y - cy

        a = np.arctan2(py, px)
        r = np.sqrt(px * px + py * py)

        sector = 2.0 * np.pi / sectors
        a = np.mod(a, sector)

        # Reflect every second sector
        reflect = a > sector * 0.5
        a = np.where(reflect, sector - a, a)

        return cx + r * np.cos(a), cy + r * np.sin(a)

    # def sample_network(self):
    #     while True:
    #         np_adj_matrix = self.sample_adj_matrix()
    #         G = nx.DiGraph(np_adj_matrix)
    #
    #         # Ensure at least one input→output path
    #         if not any(nx.has_path(G, i, o) for i in self.input_ids for o in self.output_ids):
    #             continue
    #
    #         # Ensure every hidden node lies on some input→hidden→output path
    #         all_hidden_connected = True
    #         for h in self.middle_ids:
    #             if not any(nx.has_path(G, i, h) for i in self.input_ids):
    #                 all_hidden_connected = False
    #                 break
    #             if not any(nx.has_path(G, h, o) for o in self.output_ids):
    #                 all_hidden_connected = False
    #                 break
    #
    #         if all_hidden_connected:
    #             break
    #
    #     # # Reorder nodes: inputs first, then acyclic hidden nodes, then cyclic hidden nodes, then outputs
    #     # ordered = list(self.input_ids)
    #     # available = set(self.middle_ids)
    #     #
    #     # while available:
    #     #     added = False
    #     #     for node in sorted(available):
    #     #         sources = np.where(np_adj_matrix[:, node])[0]
    #     #         if all(s in ordered for s in sources):
    #     #             ordered.append(node)
    #     #             available.remove(node)
    #     #             added = True
    #     #             break
    #     #     if not added:
    #     #         break  # cycle detected
    #     #
    #     # cyclic_start = len(ordered)
    #     # ordered.extend(sorted(available))  # cyclic hidden nodes
    #     # ordered.extend(self.output_ids)
    #
    #     # # Apply node reordering to adj matrix
    #     # reordered_adj = np.zeros_like(np_adj_matrix)
    #     # for i, old_i in enumerate(ordered):
    #     #     for j, old_j in enumerate(ordered):
    #     #         reordered_adj[i, j] = np_adj_matrix[old_i, old_j]
    #
    #     # Finalize
    #     key = random.PRNGKey(int(time.time()))
    #     jnp_adj_matrix = jnp.array(np_adj_matrix).astype(jnp.bool_)
    #     self.device_state['weights'] = jax.device_put((random.uniform(key, (self.n_nodes, self.n_nodes)) * 2 - 1) * jnp_adj_matrix).astype(jnp.float16)
    #     self.device_state['activation_ids'] = jax.device_put(jnp.array(np.random.randint(0, len(self.activations), size=self.n_hidden), dtype=jnp.int32))
    #     self.device_state['adj_matrix'] = jax.device_put(jnp_adj_matrix)
    #     # self.cyclic_start = cyclic_start
    #     self.needs_update = True
    #     self.is_valid = self.is_network_valid()
    #     print(f"[CPPN] Sampled new network")

    # def is_network_valid(self):
    #     """Fast connectivity check directly on JAX array"""
    #     # Get active node mask
    #     active_mask = self.device_state['node_active']
    #
    #     # Mask the adjacency matrix - only connections between active nodes are valid
    #     # If either source or target node is inactive, the connection is invalid
    #     masked_adj = (self.device_state['adj_matrix'] &
    #                   active_mask[:, None] &  # source node must be active
    #                   active_mask[None, :])  # target node must be active
    #
    #     # Compute transitive closure on masked adjacency matrix
    #     reachable = masked_adj > 0
    #     prev = jnp.zeros_like(reachable, dtype=jnp.bool_)
    #
    #     while not jnp.array_equal(reachable, prev):
    #         prev = reachable
    #         reachable = (reachable @ reachable) | reachable
    #
    #     # Check input -> output connectivity (only among active nodes)
    #     output_start = self.n_inputs + self.n_hidden
    #     active_inputs = active_mask[:self.n_inputs]
    #     active_outputs = active_mask[output_start:]
    #
    #     # Only check paths from active inputs to active outputs
    #     valid_paths = reachable[:self.n_inputs, output_start:] & active_inputs[:, None] & active_outputs[None, :]
    #
    #     return bool(jnp.any(valid_paths))
    #
    # def sample_adj_matrix(self):
    #     adj = np.zeros((self.n_nodes, self.n_nodes))
    #     for i in range(self.n_inputs + self.n_hidden):  # no output→anything
    #         for j in range(self.n_inputs, self.n_nodes):
    #             if i == 0 or i == 1:
    #                 continue
    #             if i in self.input_ids and j in self.output_ids:
    #                 continue
    #             # Allow cycles: no j > i restriction
    #             prob = 0.4 if j >= self.n_inputs + self.n_hidden else 0.2  # prioritize middle→output
    #             if np.random.rand() < prob:
    #                 adj[i, j] = 1
    #     return adj


    def save_state(self, viz_params=None):
        if not self.last_update_saved:
            timestamp = self.timestamp
            pkl_path = f"{self.output_path}/state_{timestamp}.pkl"
            img_path = f"{self.output_path}/image_{timestamp}.png"
            save_data = self.state
            if viz_params is not None:
                save_data['viz_params'] = viz_params
            with open(pkl_path, 'wb') as f:
                pickle.dump(save_data, f)
            del save_data
            self._x_in = None  # free 1024-res GPU tensor before high-res render
            img = self.update(res=self.high_res)  # uint8 on device
            display_image = np.array(img)
            del img

            # Apply post-processing effects if viz_params provided
            if viz_params is not None:
                self.grain_strength = viz_params.get('grain_strength', 0.0)
                self.displace_strength = viz_params.get('displace_strength', 0.0)
                self.invert = viz_params.get('invert', False)
                self.symmetry_mode = viz_params.get('symmetry_mode', 0)
                display_image = self.apply_postprocessing(display_image)

            # Flip vertically to match OpenGL display orientation
            display_image = np.flipud(display_image)

            print(f'Checkpoints saved at {img_path}')
            im = Image.fromarray(display_image)
            im.save(img_path)
            self.last_update_saved = True
            # return img

    def _inputs_key(self, res=None):
        p1 = np.array(self.device_state['input_params1'])
        p2 = np.array(self.device_state['input_params2'])
        inv = np.array(self.device_state['inverted_inputs'])
        act = np.array(self.device_state['node_active'][:self.n_inputs])
        fids = np.array(self.device_state['input_function_ids'][:self.n_inputs])
        return (
            tuple(p1.tolist()),
            tuple(p2.tolist()),
            tuple(inv.tolist()),
            tuple(act.tolist()),
            tuple(fids.tolist()),
            res,  # 👈 add resolution to invalidate cache properly
        )

    @property
    def timestamp(self):
        return time.strftime("%Y_%m_%d_%H%M%S")

    @property
    def state(self):
        return {k: np.array(v) for k, v in self.device_state.items()}

    def set_state(self, state=None, state_path=None):
        assert state != None or state_path != None
        if state is None:
            with open(state_path, "rb") as f:
                state = pickle.load(f)
        self.viz_params = state.pop('viz_params', None)
        self.device_state = {k: jnp.array(v) for k, v in state.items()}

    @property
    def time(self):
        return time.time() - self.t_start


    def print_connections(self):
        """Print adjacency matrix as readable text"""
        adj = np.array(self.device_state['adj_matrix'])
        active = np.array(self.device_state['node_active'])

        print("Network Connections:")
        print("-" * 40)

        for i in range(self.n_nodes):
            # Find all nodes that node i connects to
            connections = np.where(adj[i, :] == 1)[0]

            if len(connections) > 0:  # Skip nodes with no outgoing connections
                # Determine node type
                if i < self.n_inputs:
                    node_type = "Input"
                elif i < self.n_inputs + self.n_hidden:
                    node_type = "Hidden"
                else:
                    node_type = "Output"

                # Show active status
                status = "ACTIVE" if active[i] else "INACTIVE"

                # Format connection list
                conn_list = ", ".join([str(c+1) for c in connections])

                print(f"{node_type} {i+1} ({status}) → [{conn_list}]")

        print("-" * 40)



if __name__ == '__main__':
    import os
    RES = 1024
    FACTOR = 16 / 9

    params = dict(debug=True, res=RES, factor=FACTOR)
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

    cppn = CPPN(output_path=repo_path + 'outputs/tests/', params=params)
    cppn.sample_network()
    img = cppn.update()

    # vis = create_backend('pygame')
    vis = create_backend('moderngl')
    vis.initialize(
        cppn=cppn,
        width=int(RES * FACTOR),
        height=RES,
        window_scale=1  # Adjust this to make window bigger/smaller
    )
    vis.update(img)

    for _ in range(500):  # run more iterations
        time.sleep(0.03)  # ~30 FPS target
        t_init = time.time()

        # --- Simulate small random weight changes ---
        noise_scale = 0.02  # smaller = slower evolution
        key = random.PRNGKey(int(time.time() * 1e6) & 0xFFFFFFFF)
        noise = (random.uniform(key, cppn.device_state['weights'].shape, minval=-1.0, maxval=1.0)
                 * noise_scale).astype(cppn.device_state['weights'].dtype)
        cppn.device_state['weights'] = cppn.device_state['weights'] + noise
        cppn.device_state['weights'] *= cppn.device_state['adj_matrix'].astype(cppn.device_state['weights'].dtype)

        t_sample = time.time() - t_init

        # --- Render update ---
        t_init = time.time()
        img = cppn.update()
        jax.block_until_ready(img)  # <- make compute time explicit
        t_update = time.time() - t_init

        # --- Display ---
        t_init = time.time()
        times = vis.update(img)
        t_viz = time.time() - t_init

        print(f"Δweights: {t_sample * 1000:.2f} ms | update: {t_update * 1000:.2f} ms | viz: {t_viz * 1000:.2f} ms, times: {times}")
