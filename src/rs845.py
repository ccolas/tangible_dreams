import serial
import time
import os
from src.cppn import CPPN
import asyncio
import numpy as np

def norm_value(value, magnitude):
    return value / 1023 - 0.5 * 2 * magnitude

def exp_symmetric(value, vmin=0, vmax=1023, range_=10):
    # Normalize to [-1, 1], centered at midpoint
    norm = (value - (vmin + vmax) / 2) / ((vmax - vmin) / 2)
    # Apply symmetric exponential scaling
    scaled = np.sign(norm) * (np.exp(np.abs(norm) * 3) - 1) / (np.exp(3) - 1)
    return range_ * scaled

def exp_positive(value, vmin=0, vmax=1023, log_range=3):
    # Normalize to [-1, 1], centered at midpoint
    norm = (value - (vmin + vmax) / 2) / ((vmax - vmin) / 2)
    # Scale symmetrically around log10(1) = 0
    return 10 ** (norm * log_range)

class RS485Controller:
    def __init__(self, output_path, params):
        self.output_path = output_path
        self.params = params
        self.debug = params.get('debug', False)

        self.cppn = CPPN(output_path, params)

        # RS485 config
        self.port = params.get('port', '/dev/ttyUSB0')
        self.baud = 115200
        self.timeout = 0.004
        self.total_timeout = 0.05
        self.max_retries = 3
        # self.full_refresh_rate = 5
        self.node_ids = range(1, 20)
        self.command = 0x01  # ask for full data
        self.start_byte = 0xAA
        self.end_byte = 0xBB
        self.sync_byte = 0xCC

        # identify node ids
        self.input_ids = [1, 2, 3, 4, 5, 6]
        self.middle_ids = list(range(7, 17))
        self.output_ids = [17, 18, 19]
        self.real_nodes = [1, 7, 17]

        # param ranges
        self.weight_range = [-3, 3]
        self.ensure_low_latency()

    def ensure_low_latency(self):
        try:
            device = os.path.basename(self.port)
            latency_path = f"/sys/bus/usb-serial/devices/{device}/latency_timer"
            if os.path.exists(latency_path):
                with open(latency_path, 'r') as f:
                    current = int(f.read().strip())
                    if current != 1:
                        print(f"[WARN] latency_timer is {current}, not 1")
                        print(f"[ACTION] Run this to fix it:\n  sudo sh -c 'echo 1 > {latency_path}'")
                        assert False
                    else:
                        print(f"[INFO] latency_timer already set to 1 ms for {device}")
            else:
                print(f"[WARN] latency_timer path not found for {device}")
        except Exception as e:
            print(f"[ERROR] Failed to check latency_timer: {e}")
            assert False


    async def start_polling_loop(self):
        print('Starting RS845', flush=True)
        with serial.Serial(self.port, self.baud, timeout=self.timeout) as ser:
            await asyncio.sleep(1)

            while True:
                changes = self.poll_nodes_once(ser)
                filtered_changes = dict()
                for k, v in changes.items():
                    if k in self.real_nodes:
                        filtered_changes[k] = v
                if filtered_changes:
                    self.handle_update(filtered_changes)
                if self.command == 0x01: self.command = 0x00  # ask for udpates only
                await asyncio.sleep(0.00016)  # match main loop timing


    def start(self):
        with serial.Serial(self.port, self.baud, timeout=self.timeout) as ser:
            time.sleep(1)
            while True:
                t_init = time.time()
                changes = self.poll_nodes_once(ser)
                if changes:
                    self.handle_update(changes)
                    print(f"  took: {(time.time() - t_init)/1e3:.2f}ms", flush=True)
                if self.command == 0x01: self.command = 0x00  # ask for udpates only

    def poll_nodes_once(self, ser):
        all_changes = {}
        for node_id in self.node_ids:
            result = self.poll_single_node(ser, node_id)
            if result:
                node_id, parsed = self.parse_node_response(result)
                if parsed:
                    all_changes[node_id] = parsed
            else:
                print(f"[MISS] Node {node_id}")

        return all_changes

    def poll_single_node(self, ser, node_id):
        if self.command == 0x00:
            timeout = self.timeout
        else:
            timeout = self.total_timeout
        for _ in range(self.max_retries):
            ser.reset_input_buffer()
            ser.write(bytes([self.sync_byte, node_id, self.command]))

            buffer = bytearray()
            reading = False
            start_time = time.time()

            while time.time() - start_time < timeout:
                byte = ser.read()
                if not byte:
                    continue
                byte = byte[0]

                if byte == self.start_byte:
                    buffer = bytearray()
                    reading = True
                elif byte == self.end_byte and reading:
                    break
                elif reading:
                    buffer.append(byte)

            if len(buffer) >= 2:
                return buffer

        return None

    def parse_node_response(self, data):
        if len(data) < 2:
            return None, None

        node_id = data[0]
        flag = data[1]

        if flag == 0x01:  # no data for this node
            return node_id, {}

        if flag == 0xFF:  # some data for this node
            if len(data) < 3:
                return None, None
            change_count = data[2]  # number of values that changed
            parsed = {}
            index = 3
            for _ in range(change_count):
                if index >= len(data):
                    return None, None
                pin = data[index]
                index += 1
                if pin < 8:
                    if index + 1 >= len(data):
                        return None, None
                    value = data[index] | (data[index + 1] << 8)
                    index += 2
                elif pin < 10:
                    if index >= len(data):
                        return None, None
                    value = data[index]
                    index += 1
                else:
                    return None, None
                parsed[pin] = value
            return node_id, parsed

        return None, None


    def handle_update(self, changes):
        """Override this method to apply changes to CPPN"""
        print(changes, flush=True)

        # Nodes 1-6 are input nodes
        # Nodes 7-16 are middle nodes
        # Nodes 17-19 are output nodes

        # Sensors 0-2 are input node ids
        # Sensors 3-5 are weights on inputs
        # Sensors 6-7 are bias and slope
        # digital sensors 8-9 are switches
        for node_id, node_data in changes.items():
            if node_id in self.input_ids and node_id == 1:
                node_idx = node_id - 1
                # no inputs
                # sensor 5 (optional) is a switch on input function
                # sensor 6-7 are bias and zoom
                # digital 8 is negation switch
                # digital 9 is deactivating the node
                for sensor_id, sensor_value in node_data.items():
                    if sensor_id == 7:
                        zoom_val = exp_positive(sensor_value)
                        print(f'zoom val: {sensor_value}, {zoom_val}')
                        self.cppn.input_zooms = self.cppn.input_zooms.at[node_idx].set(zoom_val)
                    elif sensor_id == 6:
                        print(f'input func val: {sensor_value}')
                        self.cppn.input_function_ids = self.cppn.input_function_ids.at[node_idx].set(sensor_value)
                    elif sensor_id == 9:
                        print(f'input node active: {sensor_value}')
                        self.cppn.node_active = self.cppn.node_active.at[node_idx].set(int(sensor_value))
                    # if sensor_id == 5 and node_id in [5, 6]:
                    #     assert sensor_value < len(self.cppn.input_function_keys)
                    #     self.cppn.input_function_ids = self.cppn.input_function_ids.at[input_node_id].set(sensor_value)
                    # elif sensor_id == 6:
                    #     bias_val = exp_value(sensor_value)
                    #     self.cppn.input_biases = self.cppn.input_biases.at[input_node_id].set(bias_val)
                    # elif sensor_id == 7:
                    #     zoom_val = exp_value(sensor_value)
                    #     self.cppn.input_zooms = self.cppn.input_zooms.at[input_node_id].set(zoom_val)
                    # elif sensor_id == 8:
                    #     self.cppn.input_inverted = self.cppn.input_inverted.at[input_node_id].set(int(sensor_value))
                    # elif sensor_id == 9:
                    #     self.cppn.node_active = self.cppn.node_active.at[input_node_id].set(int(sensor_value))
                    # else:
                    #     raise ValueError(f"Unknown sensor {sensor_id} for input node {node_id}")



            elif node_id in self.middle_ids and node_id==7:
                node_idx = node_id - 1

                # sensors 0-2 are input ids
                # sensors 3-5 are input weights
                # sensor 6 is bias or slope
                # sensor 7 is activ
                # digital sensor 8 is switch input 1 to CV control
                # digital sensor 9 is deactivating the node
                for sensor_id, sensor_value in node_data.items():
                    if sensor_id in [0, 1]:  # Input node ids
                        source_id = int(sensor_value)
                        source_idx = source_id - 1
                        old_source_idx = self.cppn.inputs_nodes_record[node_idx, sensor_id]
                        if old_source_idx >= 0:
                            self.cppn.adj_matrix = self.cppn.adj_matrix.at[old_source_idx, node_idx].set(0)  # remove previous connection
                            self.cppn.inputs_nodes_record[node_idx, sensor_id] = -1  # udpate record
                            print(f'disconnecting {old_source_idx + 1} from {node_id}')
                        if source_idx >= 0:
                            self.cppn.inputs_nodes_record[node_idx, sensor_id] = source_idx  # keep track of connected nodes
                            self.cppn.adj_matrix = self.cppn.adj_matrix.at[source_idx, node_idx].set(1)
                            print(f'connecting {source_id} to {node_id}')
                    if sensor_id == 3:
                        slot = 0
                        input_idx = int(node_data.get(slot, -1))
                        input_idx = 0  # int(node_data.get(slot, -1))
                        self.cppn.adj_matrix = self.cppn.adj_matrix.at[input_idx, node_idx].set(1)
                        if input_idx >= 0:
                            weight = exp_symmetric(sensor_value)
                            self.cppn.weights = self.cppn.weights.at[input_idx, node_idx].set(weight)
                            if self.cppn.adj_matrix[input_idx, node_idx]:
                                print(f'weight update node {node_id}, {weight}')
                    elif sensor_id == 6:
                        bias_val = exp_symmetric(sensor_value)
                        print(f'bias val: {bias_val}')
                        self.cppn.biases = self.cppn.biases.at[node_idx].set(bias_val)
                    elif sensor_id == 9:
                        print(f'node active: {sensor_value}')
                        self.cppn.node_active = self.cppn.node_active.at[node_idx].set(int(sensor_value))
                    elif sensor_id == 7:
                        print(f'node activation function: {sensor_value}')
                        self.cppn.activation_ids = self.cppn.activation_ids.at[node_idx].set(int(sensor_value))
                # use_cv = bool(node_data.get(8, 0)) or self.cppn.cv_override[middle_node_id]
                    # if sensor_id in [0, 1, 2]:  # Input node ids
                    #     source_id = int(sensor_value)
                    #     if sensor_id == 0 and use_cv:
                    #         pass  # CV overrides this connection; don't store or set
                    #     else:
                    #         if sensor_id == 0:
                    #             self.cppn.inputs_1[middle_node_id] = source_id
                    #         if source_id >= 0:
                    #             self.cppn.adj_matrix = self.cppn.adj_matrix.at[source_id, middle_node_id].set(1)
                    # elif sensor_id in [3, 4, 5]:  # Input weights
                    #     if sensor_id == 3 and use_cv:
                    #         # Use sensor 0 value as CV weight
                    #         input_idx = int(node_data.get(1, -1))
                    #         if input_idx >= 0:
                    #             weight_val = exp_value(node_data.get(0, 0.0))
                    #             self.cppn.weights = self.cppn.weights.at[input_idx, middle_node_id].set(weight_val)
                    #     elif sensor_id == 4 and use_cv:
                    #         # this is overridden by CV on input 0
                    #         pass
                    #     else:
                    #         # Normal weight setting
                    #         slot = sensor_id - 3
                    #         input_idx = int(node_data.get(slot, -1))
                    #         if input_idx >= 0:
                    #             weight = exp_value(sensor_value)
                    #             self.cppn.weights = self.cppn.weights.at[input_idx, middle_node_id].set(weight)
                    # elif sensor_id == 6:
                    #     self.cppn.biases = self.cppn.biases.at[middle_node_id].set(exp_value(sensor_value))
                    # elif sensor_id == 7:
                    #     self.cppn.slopes = self.cppn.slopes.at[middle_node_id].set(exp_value(sensor_value))
                    # elif sensor_id == 8:
                    #     use_cv = bool(sensor_value)
                    #     self.cppn.cv_override[middle_node_id] = use_cv
                    #     if use_cv:
                    #         # Remove edge from input 0 (if it exists)
                    #         source_id = self.cppn.inputs_1[middle_node_id]
                    #         self.cppn.adj_matrix = self.cppn.adj_matrix.at[source_id, middle_node_id].set(0)
                    # elif sensor_id == 9:
                    #     self.cppn.node_active = self.cppn.node_active.at[middle_node_id].set(int(sensor_value))
                    # else:
                    #     raise ValueError
            elif node_id in self.output_ids and node_id == 17:
                # sensors 0-2 are input ids
                # sensors 3-5 are input weights
                # sensors 6-7 are bias and slope
                # digital sensor 8 is switch input 1 to CV control
                # digital sensor 9 switches between rgb and hsl
                node_idx = node_id - 1
                output_node_idx = node_id - self.output_ids[0]
                for sensor_id, sensor_value in node_data.items():
                    if sensor_id in [1]:  # Input node ids
                        source_id = int(sensor_value)
                        source_idx = source_id - 1
                        old_source_idx = self.cppn.inputs_nodes_record[node_idx, sensor_id]
                        if old_source_idx >= 0:
                            self.cppn.adj_matrix = self.cppn.adj_matrix.at[old_source_idx, node_idx].set(0)  # remove previous connection
                            self.cppn.inputs_nodes_record[node_idx, sensor_id] = -1  # udpate record
                            print(f'disconnecting {old_source_idx + 1} from {node_id}')
                        if source_idx >= 0:
                            self.cppn.inputs_nodes_record[node_idx, sensor_id] = source_idx  # keep track of connected nodes
                            self.cppn.adj_matrix = self.cppn.adj_matrix.at[source_idx, node_idx].set(1)
                            print(f'connecting {source_id} to {node_id}')

                    if sensor_id == 6:
                        slot = 0
                        input_idx = 6#int(node_data.get(slot, -1))
                        self.cppn.adj_matrix = self.cppn.adj_matrix.at[input_idx, node_idx].set(1)
                        if input_idx >= 0:
                            weight = exp_symmetric(sensor_value)
                            self.cppn.weights = self.cppn.weights.at[input_idx, node_idx].set(weight)
                            if self.cppn.adj_matrix[input_idx, node_idx]:
                                print(f'output weight update node {node_id}, {weight}')
                    elif sensor_id == 7:
                        bias = exp_symmetric(sensor_value)
                        print(f'output bias val: {bias}')
                        self.cppn.output_biases = self.cppn.output_biases.at[output_node_idx].set(bias)
                    # use_cv = bool(node_data.get(8, 0)) or self.cppn.cv_override[output_node_id]
                    #
                    # if sensor_id in [0, 1, 2]:  # Input node ids
                    #     source_id = int(sensor_value)
                    #     if sensor_id == 0 and use_cv:
                    #         pass  # CV overrides this connection; skip it
                    #     else:
                    #         if sensor_id == 0:
                    #             self.cppn.inputs_1[output_node_id] = source_id
                    #         if source_id >= 0:
                    #             self.cppn.adj_matrix = self.cppn.adj_matrix.at[source_id, output_node_id].set(1)
                    #
                    # elif sensor_id in [3, 4, 5]:  # Input weights
                    #     if sensor_id == 3 and use_cv:
                    #         input_idx = int(node_data.get(1, -1))
                    #         if input_idx >= 0:
                    #             weight_val = exp_value(node_data.get(0, 0.0))
                    #             self.cppn.weights = self.cppn.weights.at[input_idx, output_node_id].set(weight_val)
                    #     elif sensor_id == 4 and use_cv:
                    #         pass  # ignored due to CV
                    #     else:
                    #         slot = sensor_id - 3
                    #         input_idx = int(node_data.get(slot, -1))
                    #         if input_idx >= 0:
                    #             weight = exp_value(sensor_value)
                    #             self.cppn.weights = self.cppn.weights.at[input_idx, output_node_id].set(weight)
                    #
                    # elif sensor_id == 6:
                    #     self.cppn.output_biases = self.cppn.output_biases.at[output_node_id].set(exp_value(sensor_value))
                    #
                    # elif sensor_id == 7:
                    #     self.cppn.output_slopes = self.cppn.output_slopes.at[output_node_id].set(exp_value(sensor_value))
                    #
                    # elif sensor_id == 8:
                    #     use_cv = bool(sensor_value)
                    #     self.cppn.cv_override[output_node_id] = use_cv
                    #     if use_cv:
                    #         source_id = self.cppn.inputs_1[output_node_id]
                    #         if source_id >= 0:
                    #             self.cppn.adj_matrix = self.cppn.adj_matrix.at[source_id, output_node_id].set(0)
                    #
                    # elif sensor_id == 9:
                    #     self.cppn.output_modes = self.cppn.output_modes.at[output_node_id].set(int(sensor_value))
                    #
                    # else:
                    #     raise ValueError(f"Unknown sensor {sensor_id} for output node {node_id}")
        if len(changes) > 0:
            self.cppn.needs_update = True

if __name__ == '__main__':
    DEBUG = False
    USE_AUDIO = False
    RES = 1024
    FACTOR = 16 / 9

    params = dict(debug=DEBUG, res=RES, factor=FACTOR, use_audio=USE_AUDIO)
    controller = RS485Controller('~/', params)
    controller.start()
