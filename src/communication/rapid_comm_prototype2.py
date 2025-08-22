import serial
import time
from collections import defaultdict
import os

# === CONFIG ===
PORT = '/dev/ttyUSB0'
BAUD = 115200
START_BYTE = 0xAA
END_BYTE = 0xBB
TIMEOUT = 0.004  # 4ms
TIMEOUT_TOTAL = 0.5
MAX_RETRIES = 3
RUNS = 3000000
NODES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
NODES_TO_PLOT = NODES


# dmesg | grep tty
def ensure_low_latency(port_path):
    try:
        device = os.path.basename(port_path)
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
ensure_low_latency(PORT)

# === PARSE NODE RESPONSE ===
def parse_node_response(data):
    if len(data) < 2:
        return None

    node_id = data[0]
    flag = data[1]

    if flag == 0x01:
        # print(f'{node_id}: no change received')
        return node_id, {}

    if flag == 0xFF:
        if len(data) < 3:
            return None

        change_count = data[2]
        parsed_data = {}
        index = 3

        for _ in range(change_count):
            if index >= len(data):
                return None  # incomplete pin block

            pin = data[index]
            index += 1

            if pin < 8:
                if index + 1 >= len(data):
                    return None
                value = data[index] | (data[index + 1] << 8)
                index += 2
            elif pin < 10:
                if index >= len(data):
                    return None
                value = data[index]
                index += 1
            else:
                return None  # invalid pin ID

            parsed_data[pin] = value

        return node_id, parsed_data

    return None



def poll_single_node(ser, node_id, command, timeout):
    t_start = time.time()
    ser.reset_input_buffer()
    ser.write(bytes([0xCC, node_id, command]))
    t_sent = time.time()
    buffer = bytearray()
    reading = False
    start_time = time.time()

    while time.time() - start_time < timeout:
        byte = ser.read()
        if not byte:
            continue
        byte = byte[0]

        if byte == START_BYTE:
            buffer = bytearray()
            reading = True
        elif byte == END_BYTE and reading:
            break
        elif reading:
            buffer.append(byte)
    t_received = time.time()
    # print(f"Node {node_id} - TX: {(t_sent-t_start)*1000:.2f}ms, RX: {(t_received-t_sent)*1000:.2f}ms")
    if len(buffer) >= 2:
        return buffer  # buffer includes: [node_id, flag, payload...]
    return None

# === POLL ALL NODES ===
def poll_nodes():
    PARSED_ONCE = False
    with serial.Serial(PORT, BAUD,  timeout=0, write_timeout=0) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        ser.inter_byte_timeout = None
        time.sleep(0.1)  # Give Arduino time to settle
        run = 0
        while True:
            run += 1
            # print(f"\n=== Simulation Run {run} ===")
            node_data = {}
            missed_nodes = []
            timings = defaultdict(float)
            command = 0x01 if (run == 1 or not PARSED_ONCE) else 0x00

            if command == 0x00:
                timeout = TIMEOUT
                max_retries = MAX_RETRIES
            else:
                timeout = TIMEOUT_TOTAL
                max_retries = 10
            i_attempts = []
            for node_id in NODES:
                i_attempt = 0
                parsed = None
                while i_attempt <= max_retries:
                    i_attempt += 1
                    t_start = time.time()
                    buffer = poll_single_node(ser, node_id, command, timeout)
                    t_read = time.time()
                    timings['read'] += (t_read - t_start) * 1000

                    if buffer:
                        parsed = parse_node_response(buffer)
                        t_parse = time.time()
                        timings['parse'] += (t_parse - t_read) * 1000
                        if parsed:
                            # if len(parsed[1]) > 0:
                            break
                        # print(f'failed to parse node {node_id} → {buffer.hex()}')
                        continue
                    # print(f'failed to read from node {node_id}')

                if parsed:
                    node_data[node_id] = parsed[1]
                    PARSED_ONCE = True
                else:
                    missed_nodes.append(node_id)
                    print(f"[FAIL] Node {node_id} (after {max_retries} attempts)")
                i_attempts.append(i_attempt)

            # print("\n[Node Data]")
            any_change = False
            for node_id, i_attempt in zip(NODES, i_attempts):
                if node_id in node_data:
                    if node_id in NODES_TO_PLOT:
                        if len(node_data[node_id]) > 0:
                            any_change = True
                            print(f"  Node {node_id}: {node_data[node_id]}; i_attempts={i_attempt}")
                            # filtered = {k: v for k, v in node_data[node_id].items() if k in [0, 1, 2]}
                            # print(f"  Node {node_id}: {filtered}")
                else:
                    print(f"  Node {node_id}: ❌")
            # if any_change:
            total_time = sum(timings.values())
            if any_change:
                print(f"[Timing] Total: {total_time:.2f} ms")
                # for k, v in timings.items():
                #     print(f"  {k}: {v:.2f} ms")

poll_nodes()
