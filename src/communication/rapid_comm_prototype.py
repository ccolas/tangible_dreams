import serial
import time
from collections import defaultdict

PORT = '/dev/ttyUSB0'
BAUD = 115200
START_BYTE = 0xAA
END_BYTE = 0xBB
TIMEOUT = 0.004  # 5ms
MAX_RETRIES = 3  # Maximum number of retries
RUNS = 5  # Number of simulation runs

import os

def ensure_low_latency(port_path):
    """Ensure the USB latency_timer is set to 1ms for low-latency serial I/O."""
    try:
        # Get the device name (e.g., 'ttyUSB0')
        device = os.path.basename(port_path)

        # Build path to latency_timer
        latency_path = f"/sys/bus/usb-serial/devices/{device}/latency_timer"

        if not os.path.exists(latency_path):
            print(f"[WARN] latency_timer path not found for {device}")
            return

        with open(latency_path, 'r') as f:
            current = int(f.read().strip())

        if current != 1:
            with open(latency_path, 'w') as f:
                f.write('1')
            print(f"[INFO] Set latency_timer to 1 ms for {device}")
        else:
            print(f"[INFO] latency_timer already set to 1 ms for {device}")

    except Exception as e:
        print(f"[ERROR] Failed to set latency_timer for {port_path}: {e}")
ensure_low_latency(PORT)

def parse_node_response(data):
    """ Parse the response from a node """
    # print(f"\n[DEBUG] Raw data received: {data.hex()}")

    if len(data) < 2:
        # print(f"[ERROR] Invalid data length: {len(data)}, data: {data.hex()}")
        return None

    node_id = data[0]
    flag = data[1]

    # print(f"[DEBUG] Node ID: {node_id}, Flag: {flag}")

    if flag == 0x01:
        # print(f"[INFO] No changes for Node {node_id}")
        return node_id, {}

    if flag == 0xFF:
        if len(data) < 3:
            # print(f"[ERROR] Incomplete change data from Node {node_id}: {data.hex()}")
            return None

        change_count = data[2]
        parsed_data = {}
        index = 3

        # print(f"[DEBUG] Change count: {change_count}")

        for _ in range(change_count):
            if index >= len(data):
                # print(f"[ERROR] Incomplete change data for Node {node_id}")
                return None

            pin = data[index]
            index += 1
            # print(f"[DEBUG] Parsing pin: {pin}, Index now at: {index}")

            # Detect if it's analog or digital
            if pin < 8:  # Analog pin
                # print(f"[DEBUG] Detected analog pin: {pin}")
                if index + 1 >= len(data):
                    # print(f"[ERROR] Incomplete analog data for Node {node_id}")
                    return None
                low_byte = data[index]
                high_byte = data[index + 1]
                value = low_byte | (high_byte << 8)
                parsed_data[pin] = value
                index += 2
                # print(f"[DEBUG] Analog pin {pin} value: {value}")

            else:  # Digital pin (8 or 9)
                # print(f"[DEBUG] Detected digital pin: {pin}")
                if index >= len(data):
                    # print(f"[ERROR] Incomplete digital data for Node {node_id}")
                    return None
                value = data[index]
                parsed_data[pin] = value
                index += 1
                # print(f"[DEBUG] Digital pin {pin} value: {value}")

        # print(f"[DEBUG] Final parsed data for Node {node_id}: {parsed_data}")
        return node_id, parsed_data

    # print(f"[ERROR] Unknown flag {flag} from Node {node_id}")
    return None

def poll_single_node(ser, node_id, command=0x00):
    """ Poll a single node with retry logic and detailed timing """
    attempt = 0

    while attempt < MAX_RETRIES:
        timing_debug = {}
        t0 = time.time()

        ser.reset_input_buffer()
        # ser.write(bytes([node_id]))
        ser.write(bytes([node_id, command]))

        t1 = time.time()

        buffer = bytearray()
        reading = False

        start_time = time.time()
        t_read_loop_start = start_time

        while time.time() - start_time < TIMEOUT:
            t_loop = time.time()
            byte = ser.read()
            t_after_read = time.time()

            timing_debug.setdefault("total_reads", 0)
            timing_debug["total_reads"] += 1
            timing_debug.setdefault("read_time_total", 0)
            timing_debug["read_time_total"] += (t_after_read - t_loop)

            if not byte:
                continue
            byte = byte[0]

            if byte == START_BYTE:
                buffer = bytearray()
                reading = True
                timing_debug["start_byte_detected"] = t_after_read - t_read_loop_start
            elif byte == END_BYTE and reading:
                timing_debug["end_byte_detected"] = t_after_read - t_read_loop_start
                break
            elif reading:
                buffer.append(byte)

        t2 = time.time()

        # if len(buffer) > 0:
        #     print(f"[DEBUG] Node {node_id} timing (ms): write={1000*(t1-t0):.2f}, "
        #           f"read_loop={1000*(t2-t1):.2f}, reads={timing_debug.get('total_reads',0)}, "
        #           f"read_total_time={1000*timing_debug.get('read_time_total',0):.2f}, "
        #           f"start_at={1000*timing_debug.get('start_byte_detected',0):.2f}, "
        #           f"end_at={1000*timing_debug.get('end_byte_detected',0):.2f}")
        #     return buffer

        attempt += 1

    return None



def poll_nodes():
    with serial.Serial(PORT, BAUD, timeout=TIMEOUT) as ser:
        time.sleep(1)
        for i_poll in range(10):
            """ Poll all nodes, parse their responses and profile timings """
            node_data = {}
            timings = defaultdict(float)
            missed_nodes = []


            command = 0x01 if i_poll == 0 else 0x00
            for node_id in range(1, 21):
                t_start = time.time()

                # Poll the node
                buffer = poll_single_node(ser, node_id, command)
                t_read = time.time()

                # Parse the response
                if buffer is None:
                    # print(f"[WARN] Missed response from Node {node_id}")
                    missed_nodes.append(node_id)
                    continue

                parsed = parse_node_response(buffer)
                if parsed:
                    node_id, changes = parsed
                    node_data[node_id] = changes
                else:
                    # print(f"[ERROR] Failed to parse response from Node {node_id}: {buffer.hex()}")
                    missed_nodes.append(node_id)

                t_parse = time.time()

                # Store timings
                timings['read_response'] += (t_read - t_start) * 1000
                timings['parse_response'] += (t_parse - t_read) * 1000

            total_time = sum(timings.values())

            # Final report
            print("\n=== Profiling Report ===")
            print(f"Total time: {total_time:.2f} ms")
            for step, duration in timings.items():
                print(f"{step}: {duration:.2f} ms ({(duration / total_time) * 100:.1f}%)")

            print("\n=== Final Node Data ===")
            for node in range(1, 21):
                if node in node_data:
                    if len(node_data[node])>0:
                        print(f"Node {node}: {node_data[node]}")
                else:
                    print(f"Node {node}: ❌ Missed")

            if len(missed_nodes) == 0:
                print("\n✅ All nodes successfully parsed!")
            else:
                print(f"\n❌ Missed Nodes: {missed_nodes}")


poll_nodes()
