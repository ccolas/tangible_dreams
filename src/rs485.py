import serial
import time
import os
import asyncio
import signal
import atexit
import numpy as np
from jax import numpy as jnp

from src.cppn import CPPN
from src.cppn_utils import weight_mapping, slope_mapping, zoom_mapping, bias_mapping, mods_mapping, balance_mapping, contrast_mapping


class RS485Controller:
    def __init__(self, output_path, params):
        self.output_path = output_path
        self.params = params
        self.debug = params.get('debug', False)

        self.cppn = CPPN(output_path, params)

        # RS485 config
        self.port = params.get('port', '/dev/ttyUSB0')
        self.baud = 115200
        self.update_timeout = 0.004
        self.total_timeout = 0.3
        self.update_max_retries = 5

        broken_nodes = []
        self.node_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        self.command_sent_with_last_update = [None for _ in range(len(self.node_ids))]
        self.last_change_time = time.time()
        self.start_byte = 0xAA
        self.end_byte = 0xBB
        self.sync_byte = 0xCC

        # identify node ids
        self.input_ids = sorted(set([1, 2, 3, 4, 5]) - set(broken_nodes))
        self.middle_ids = sorted(set([6, 7, 8, 9, 10, 11, 12, 13, 14]) - set(broken_nodes))
        self.output_ids = sorted(set([15, 16, 17]) - set(broken_nodes))
        self.node_ids = sorted(set(self.node_ids) - set(broken_nodes))

        self.weight_range = [-3, 3]
        self.ensure_low_latency()
        self.viz = None
        self.parsed_once = False

        # Serial handle (lazy open)
        self.ser = None
        self._install_clean_exit()

    def ensure_low_latency(self):
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
            assert False

    # --- Safe open/close ---
    def _open_serial(self):
        ser = serial.Serial(
            self.port, self.baud,
            timeout=0, write_timeout=0,
            rtscts=False, dsrdtr=False, xonxoff=False
        )
        ser.reset_input_buffer();
        ser.reset_output_buffer()
        t0 = time.time()
        while time.time() - t0 < 0.05:
            ser.read(4096)
        ser.write(b'\xFF' * 64)  # idle preamble
        ser.flush()
        time.sleep(0.003)
        try:
            ser.setDTR(False);
            ser.setRTS(False)
        except Exception:
            pass
        return ser

    def _close_serial(self):
        if self.ser is None:
            return
        try:
            self.ser.write(b'\xFF' * 64)
            self.ser.flush()
            time.sleep(0.003)
            self.ser.reset_input_buffer();
            self.ser.reset_output_buffer()
            try:
                self.ser.setDTR(False);
                self.ser.setRTS(False)
            except Exception:
                pass
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass
        self.ser = None

    def _install_clean_exit(self):
        atexit.register(self._close_serial)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: (_ for _ in ()).throw(SystemExit))

    # --- Polling ---
    async def start_polling_loop(self):
        print('Starting RS485', flush=True)
        self.ser = self._open_serial()
        self.ser.inter_byte_timeout = 0
        await asyncio.sleep(0.1)
        while True:
            t_init = time.time()
            if self.viz:
                self.viz.poll_events()
            changes, i_attempts = self.poll_nodes(self.ser)
            t_poll = time.time()
            self.handle_update(changes)
            t_update = time.time()
            if len(changes) > 0:
                self.last_change_time = time.time()
                if not self.parsed_once: self.parsed_once = True
                time_poll = (t_poll - t_init) * 1000
                time_update = (t_update - t_poll) * 1000
                # print(f'control time: poll={time_poll:.2f}ms, update={time_update:.2f}')
            await asyncio.sleep(0.00016)

    def start(self):
        self.ser = self._open_serial()
        self.ser.inter_byte_timeout = 0
        time.sleep(0.1)
        while True:
            t_init = time.time()
            changes, i_attempts = self.poll_nodes(self.ser)
            self.handle_update(changes)
            print(f"  took: {(time.time() - t_init)*1000:.2f}ms, attempts:{i_attempts}", flush=True)
            if len(changes) > 0 and not self.parsed_once:
                self.parsed_once = True

    def poll_nodes(self, ser):
        all_changes = {}
        i_attempts = []
        for node_id in self.node_ids:
            i_attempt = 0
            parsed = None
            if not self.parsed_once:
                command = 0x01
                timeout = self.total_timeout
                max_retries = 20
            elif (time.time() - self.last_change_time > 1 and
                  self.command_sent_with_last_update[node_id - 1] == 0x00 and
                  (node_id - 1) >= self.cppn.n_inputs):
                command = 0x02
                timeout = self.total_timeout
                max_retries = self.update_max_retries
            else:
                command = 0x00
                timeout = self.update_timeout
                max_retries = self.update_max_retries
            while i_attempt <= max_retries:
                i_attempt += 1
                buffer = self.poll_single_node(ser, node_id, command, timeout)
                if buffer:
                    _, parsed = self.parse_node_response(buffer, command)
                    if parsed is not None and (command == 0x00 or len(parsed) > 0):
                        break
                if command == 0x01 and timeout < 1.0:
                    timeout += 0.05
            if parsed is not None:
                if len(parsed) > 0:
                    all_changes[node_id] = parsed
                    self.command_sent_with_last_update[node_id - 1] = command
                    if command == 0x01:
                        print(f"[FIRST READ] Node {node_id}, {parsed}")
            else:
                print(f"[FAIL] Node {node_id} (after {max_retries} attempts)")
            i_attempts.append(i_attempt)
        return all_changes, i_attempts

    def poll_single_node(self, ser, node_id, command, timeout):
        ser.reset_input_buffer()
        if command == 0x02:
            command = 0x01
        time.sleep(0.0001)
        ser.write(bytes([self.sync_byte, node_id, command]))
        ser.flush()
        time.sleep(0.0002)

        buffer = bytearray()
        reading = False
        start_time = time.time()
        while time.time() - start_time < timeout:
            byte = ser.read()
            if not byte:
                continue
            byte = byte[0]
            if byte == self.start_byte:
                buffer = bytearray(); reading = True
            elif byte == self.end_byte and reading:
                break
            elif reading:
                buffer.append(byte)
        if len(buffer) >= 2:
            return buffer
        return None

    def parse_node_response(self, data, command):
        if len(data) < 2:
            return None, None
        node_id = data[0]
        flag = data[1]
        if flag == 0x01:
            return node_id, {}
        if flag == 0xFF:
            if len(data) < 3:
                return None, None
            change_count = data[2]
            parsed = {}
            index = 3
            for _ in range(change_count):
                if index >= len(data):
                    return None, None
                pin = data[index]; index += 1
                if pin < 8:
                    if index + 1 >= len(data):
                        return None, None
                    value = data[index] | (data[index + 1] << 8)
                    index += 2
                elif pin < 10:
                    if index >= len(data):
                        return None, None
                    value = data[index]; index += 1
                else:
                    return None, None
                if command != 0x02 or pin < 3:
                    parsed[pin] = value
            return node_id, parsed
        return None, None


    def handle_update(self, changes):
        """Override this method to apply changes to CPPN"""
        # print(changes, flush=True)

        # Nodes 1-5 are input nodes
        # Nodes 6-14 are middle nodes
        # Nodes 15-17 are output nodes
        graph_changed = False
        for node_id, node_data in changes.items():
            print(f'updating node {node_id}')

            # # # # # # # # # # # # # # #
            # INPUT CONTROL
            # # # # # # # # # # # # # # #

            # print(np.array(self.cppn.device_state['adj_matrix']).astype(int))
            if node_id in self.input_ids:
                node_idx = node_id - 1
                print(f'  node idx={node_idx}, input_idx={node_idx}')
                # no inputs
                # sensor 3 (optional) is a switch on input function (source nodes)
                # digital 9 (optional) is an "invert" on Cartesian coordinates (X, Y nodes)
                # sensor 6-7 are zoom and shift (or two mods)
                # digital 8 is on/off switch
                for sensor_id, sensor_value in node_data.items():
                    if sensor_id == 3:
                        input_function_id = sensor_value + 2
                        if 2 <= input_function_id < len(self.cppn.input_function_names):
                            self.cppn.device_state['input_function_ids'] = self.cppn.device_state['input_function_ids'].at[node_idx].set(input_function_id)
                            print(f'  set coordinate to {self.cppn.input_function_names[input_function_id]}')

                    elif sensor_id == 6:
                        self.cppn.device_state['input_params1'] = self.cppn.device_state['input_params1'].at[node_idx].set(sensor_value)
                        print(f'  mod 1: {sensor_value}')

                    elif sensor_id == 7:
                        self.cppn.device_state['input_params2'] = self.cppn.device_state['input_params2'].at[node_idx].set(sensor_value)
                        print(f'  mod 2: {sensor_value}')

                    elif sensor_id == 9:
                        print(f'  invert: {bool(sensor_value)}')
                        self.cppn.device_state['inverted_inputs'] = self.cppn.device_state['inverted_inputs'].at[node_idx].set(bool(sensor_value))

                    elif sensor_id == 8:
                        print(f'  active: {sensor_value}')
                        self.cppn.device_state['node_active'] = self.cppn.device_state['node_active'].at[node_idx].set(bool(sensor_value))
                        graph_changed = True
                    # else:
                    #     raise ValueError(f'Unexepceted sensor {sensor_id} on node {node_id}')


            # # # # # # # # # # # # # # #
            # MIDDLE CONTROL
            # # # # # # # # # # # # # # #

            elif node_id in self.middle_ids:
                node_idx = node_id - 1
                middle_idx = node_idx - len(self.input_ids)
                print(f'  node idx={node_idx}, middle_idx={middle_idx}')

                # sensors 0-2 are input ids
                # sensors 3-5 are input weights
                # sensor 6 is slope
                # sensor 7 is activ
                # digital sensor 9 is cv switch
                # digital sensor 10 is on/off switch

                # CV override first
                cv_override_update = node_data.get(9, None)
                if cv_override_update == 1 and not self.cppn.device_state['cv_override'][node_idx]:
                    print(f'  switching cv override ON for node {node_id}')
                    self.cppn.device_state['cv_override'] = self.cppn.device_state['cv_override'].at[node_idx].set(True)

                elif cv_override_update == 0 and self.cppn.device_state['cv_override'][node_idx]:
                    print(f'  switching cv override OFF for node {node_id}')
                    self.cppn.device_state['cv_override'] = self.cppn.device_state['cv_override'].at[node_idx].set(False)
                    self.cppn.device_state['weight_mods'] = self.cppn.device_state['weight_mods'].at[:, node_idx].set(0)

                # # If CV override is on, adjust connection for input 2
                # if self.cppn.device_state['cv_override'][node_idx]:
                #     sensor_id = 1
                #     input2_update = node_data.get(sensor_id, None)
                #     if input2_update is not None:
                #         source_id = input2_update
                #         source_idx = source_id - 1
                #         old_source_idx = self.cppn.inputs_nodes_record[node_idx, sensor_id]
                #         if source_idx != old_source_idx:
                #             if source_idx < len(self.node_ids):
                #                 if old_source_idx >= 0:
                #                     self.cppn.device_state['adj_matrix'] = self.cppn.device_state['adj_matrix'].at[old_source_idx, node_idx].set(False)
                #                     self.cppn.inputs_nodes_record[node_idx, sensor_id] = -1
                #                     print(f'  disconnecting {old_source_idx + 1} from {node_id}')
                #                 if source_idx >= 0:
                #                     self.cppn.inputs_nodes_record[node_idx, sensor_id] = source_idx
                #                     self.cppn.device_state['adj_matrix'] = self.cppn.device_state['adj_matrix'].at[source_idx, node_idx].set(True)
                #                     print(f'  connecting {source_id} to {node_id}')
                #             else:
                #                 print(f'  WARNING: attempt connection from {source_idx} (not a node!) to {node_id}')

                for sensor_id, sensor_value in node_data.items():
                    if sensor_id in [0, 1, 2]:  # Input node ids
                        # if self.cppn.cv_override[node_idx] and sensor_id == 0:
                        #     print(f'  reading cv signal change: {sensor_value}')
                        #     # this value becomes the weight of the connection from input 2
                        #     weight2 = weight_mapping(sensor_value)
                        #     controlled_input_id = 1
                        #     source2_idx = self.cppn.inputs_nodes_record[node_idx, controlled_input_id]
                        #     if source2_idx >= 0:
                        #         self.cppn.weight_mods = self.cppn.weight_mods.at[source2_idx, node_idx].set(weight2)
                        #         print(f'  cv controls weight 2: {weight2}')
                        # else:
                        source_id = sensor_value
                        source_idx = source_id - 1
                        old_source_idx = self.cppn.inputs_nodes_record[node_idx, sensor_id]
                        if source_idx != old_source_idx:
                            if source_idx < len(self.node_ids):
                                if old_source_idx >= 0:
                                    unique_connection = len(np.argwhere(self.cppn.inputs_nodes_record[node_idx] == old_source_idx)) == 1
                                    if unique_connection:
                                        self.cppn.device_state['adj_matrix'] = self.cppn.device_state['adj_matrix'].at[old_source_idx, node_idx].set(False)
                                        graph_changed = True
                                        print(f'  disconnecting {old_source_idx + 1} from {node_id}')
                                    self.cppn.inputs_nodes_record[node_idx, sensor_id] = -1
                                if source_idx >= 0:
                                    self.cppn.inputs_nodes_record[node_idx, sensor_id] = source_idx
                                    self.cppn.device_state['adj_matrix'] = self.cppn.device_state['adj_matrix'].at[source_idx, node_idx].set(True)
                                    graph_changed = True
                                    print(f'  connecting {source_id} to {node_id}')
                            else:
                                print(f'  WARNING: attempt connection from {source_idx} (not a node!) to {node_id}')
                    elif sensor_id in [3, 4, 5]:
                        # if sensor_id == 3 and self.cppn.cv_override[node_idx]:
                        #     #TODO we may want to use this in some way
                        #     # weight 1 (sensor 3) is deactivated (no input, cv instead)
                        #     pass
                        # else:
                        input_idx = sensor_id - 3
                        source_idx = self.cppn.inputs_nodes_record[node_idx, input_idx]
                        if source_idx >= 0:
                            weight = weight_mapping(sensor_value)
                            self.cppn.device_state['weights'] = self.cppn.device_state['weights'].at[source_idx, node_idx].set(weight)
                            if self.cppn.device_state['adj_matrix'][source_idx, node_idx]:
                                print(f'  update weight {input_idx + 1} modulating source {source_idx + 1}: {weight}')
                        self.cppn.weights_1_raw[node_idx] = sensor_value


                    elif sensor_id == 7:
                        activ_id = sensor_value
                        if 0 <= activ_id < len(self.cppn.device_state['activation_ids']):
                            self.cppn.device_state['activation_ids'] = self.cppn.device_state['activation_ids'].at[middle_idx].set(activ_id)
                            print(f'  updating activation to {self.cppn.activations_names[activ_id]}')
                    elif sensor_id == 6:
                        slope = slope_mapping(sensor_value)
                        self.cppn.device_state['slopes'] = self.cppn.device_state['slopes'].at[middle_idx].set(slope)
                        print(f'  changing slope value: {slope}')
                    elif sensor_id == 8:
                        self.cppn.device_state['node_active'] = self.cppn.device_state['node_active'].at[node_idx].set(bool(sensor_value))
                        graph_changed = True
                        if sensor_value:
                            print('  activating node')
                        else:
                            print('  deactivating node')
                    elif sensor_id == 9:
                        pass
                    # else:
                    #     raise ValueError(f'Unexpected sensor {sensor_id} on node {node_id}')


            # # # # # # # # # # # # # # #
            # OUTPUT CONTROL
            # # # # # # # # # # # # # # #

            elif node_id in self.output_ids:

                node_idx = node_id - 1
                output_idx = node_id - self.output_ids[0]
                print(f'  node idx={node_idx}, output_node_idx={output_idx}')

                # sensors 0-2 are input ids
                # sensors 3-5 are input weights
                # sensor 6 is balance (bias)
                # sensor 7 is contrast (slope)
                # digital sensor 9 is cv switch
                # digital sensor 10 is on/off switch

                # CV override first
                cv_override_update = node_data.get(9, None)
                if cv_override_update == 1 and not self.cppn.device_state['cv_override'][node_idx]:
                    print(f'  switching cv override ON for node {node_id}')
                    self.cppn.device_state['cv_override'] = self.cppn.device_state['cv_override'].at[node_idx].set(True)

                elif cv_override_update == 0 and self.cppn.device_state['cv_override'][node_idx]:
                    print(f'  switching cv override OFF for node {node_id}')
                    self.cppn.device_state['cv_override'] = self.cppn.device_state['cv_override'].at[node_idx].set(False)
                    self.cppn.device_state['weight_mods'] = self.cppn.device_state['weight_mods'].at[:, node_idx].set(0)

                # if self.cppn.device_state['cv_override'][node_idx]:
                #     sensor_id = 1
                #     input2_update = node_data.get(sensor_id, None)
                #     if input2_update is not None:
                #         source_id = input2_update
                #         source_idx = source_id - 1
                #         old_source_idx = self.cppn.inputs_nodes_record[node_idx, sensor_id]
                #         if source_idx != old_source_idx:
                #             if source_idx < len(self.node_ids):
                #                 if old_source_idx >= 0:
                #                     self.cppn.device_state['adj_matrix'] = self.cppn.device_state['adj_matrix'].at[old_source_idx, node_idx].set(False)
                #                     self.cppn.inputs_nodes_record[node_idx, sensor_id] = -1
                #                     print(f'  disconnecting {old_source_idx + 1} from {node_id}')
                #                 if source_idx >= 0:
                #                     self.cppn.inputs_nodes_record[node_idx, sensor_id] = source_idx
                #                     self.cppn.device_state['adj_matrix'] = self.cppn.device_state['adj_matrix'].at[source_idx, node_idx].set(True)
                #                     print(f'  connecting {source_id} to {node_id}')
                #             else:
                #                 print(f'  WARNING: attempt connection from {source_idx} (not a node!) to {node_id}')

                for sensor_id, sensor_value in node_data.items():
                    if sensor_id in [0, 1, 2]:  # Input node ids
                    #     if self.cppn.device_state['cv_override'][node_idx] and sensor_id == 0:
                    #         print(f'  reading cv signal change: {sensor_value}')
                    #         # this value becomes the weight of the connection from input 2
                    #         weight2 = weight_mapping(sensor_value)
                    #         controlled_input_id = 1
                    #         source2_idx = self.cppn.inputs_nodes_record[node_idx, controlled_input_id]
                    #         if source2_idx >= 0:
                    #             self.cppn.weight_mods = self.cppn.weight_mods.at[source2_idx, node_idx].set(weight2)
                    #             print(f'  cv controls weight 2: {weight2}')
                    #     else:
                        source_id = sensor_value
                        source_idx = source_id - 1
                        old_source_idx = self.cppn.inputs_nodes_record[node_idx, sensor_id]
                        if source_idx != old_source_idx:
                            if source_idx < len(self.node_ids):
                                if old_source_idx >= 0:
                                    unique_connection = len(np.argwhere(self.cppn.inputs_nodes_record[node_idx] == old_source_idx)) == 1
                                    if unique_connection:
                                        self.cppn.device_state['adj_matrix'] = self.cppn.device_state['adj_matrix'].at[old_source_idx, node_idx].set(False)
                                        graph_changed = True
                                        print(f'  disconnecting {old_source_idx + 1} from {node_id}')
                                    self.cppn.inputs_nodes_record[node_idx, sensor_id] = -1
                                if source_idx >= 0:
                                    self.cppn.inputs_nodes_record[node_idx, sensor_id] = source_idx
                                    self.cppn.device_state['adj_matrix'] = self.cppn.device_state['adj_matrix'].at[source_idx, node_idx].set(True)
                                    graph_changed = True
                                    print(f'  connecting {source_id} to {node_id}')
                            else:
                                print(f'  WARNING: attempt connection from {source_idx} (not a node!) to {node_id}')

                    elif sensor_id in [3, 4, 5]:
                        # if sensor_id == 3 and self.cppn.cv_override[node_idx]:
                        #     # TODO we may want to use this in some way
                        #     # weight 1 (sensor 3) is deactivated (no input, cv instead)
                        #     pass
                        # else:
                        input_idx = sensor_id - 3
                        source_idx = self.cppn.inputs_nodes_record[node_idx, input_idx]
                        if source_idx >= 0:
                            weight = weight_mapping(sensor_value)
                            self.cppn.device_state['weights'] = self.cppn.device_state['weights'].at[source_idx, node_idx].set(weight)
                            if self.cppn.device_state['adj_matrix'][source_idx, node_idx]:
                                print(f'  update weight {input_idx + 1} modulating source {source_idx + 1}: {sensor_value}, {weight}')
                        self.cppn.weights_1_raw[node_idx] = sensor_value
                    elif sensor_id == 6:
                        bias = balance_mapping(sensor_value)
                        self.cppn.device_state['output_biases'] = self.cppn.device_state['output_biases'].at[output_idx].set(bias)
                        print(f'  changing balance value: {bias}')
                    elif sensor_id == 7:
                        slope = contrast_mapping(sensor_value)
                        self.cppn.device_state['output_slopes'] = self.cppn.device_state['output_slopes'].at[output_idx].set(slope)
                        print(f'  changing contrast value: {slope}')
                    elif sensor_id == 8:
                        self.cppn.device_state['node_active'] = self.cppn.device_state['node_active'].at[node_idx].set(bool(sensor_value))
                        graph_changed = True
                        if sensor_value:
                            print('  activating node')
                        else:
                            print('  deactivating node')
                    elif sensor_id == 9:
                        pass
                    # else:
                    #     raise ValueError(f'Unexepceted sensor {sensor_id} on node {node_id}')

        if graph_changed:
            self.cppn.update_cycles_if_needed()


        if len(changes) > 0:
            self.cppn.needs_update = True

        # update cv
        for node_idx in range(self.cppn.n_inputs, self.cppn.n_nodes):
            if self.cppn.device_state['cv_override'][node_idx]:
                weight1 = self.cppn.weights_1_raw[node_idx]

                # update weight of second input
                weight2 = self.cppn.reactive_update(node_idx, weight1)
                if weight2 is not None:
                    weight2 = jnp.asarray(weight2, dtype=jnp.float16)
                    # find node connected in input 2
                    source2_idx = self.cppn.inputs_nodes_record[node_idx, 1]
                    if source2_idx >= 0:
                        self.cppn.device_state['weight_mods'] = self.cppn.device_state['weight_mods'].at[source2_idx, node_idx].set(weight2)
                        # print(f'  cv controls weight 2: {weight2}')
                    self.cppn.needs_update = True


if __name__ == '__main__':
    DEBUG = False
    USE_AUDIO = False
    RES = 1024
    FACTOR = 16 / 9

    params = dict(debug=DEBUG, res=RES, factor=FACTOR, use_audio=USE_AUDIO)
    controller = RS485Controller('~/', params)
    controller.start()
