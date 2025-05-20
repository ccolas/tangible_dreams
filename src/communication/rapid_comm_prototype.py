import serial
import time
from collections import defaultdict

PORT = '/dev/ttyUSB4'
BAUD = 115200
START_BYTE = 0xAA
END_BYTE = 0xBB
TIMEOUT = 0.004  # 5ms
MAX_RETRIES = 3  # Maximum number of retries
RUNS = 5  # Number of simulation runs


def parse_node_response(data):
    """ Parse the response from a node """
    print(f"\n[DEBUG] Raw data received: {data.hex()}")

    if len(data) < 2:
        print(f"[ERROR] Invalid data length: {len(data)}, data: {data.hex()}")
        return None

    node_id = data[0]
    flag = data[1]

    print(f"[DEBUG] Node ID: {node_id}, Flag: {flag}")

    if flag == 0x01:
        print(f"[INFO] No changes for Node {node_id}")
        return node_id, {}

    if flag == 0xFF:
        if len(data) < 3:
            print(f"[ERROR] Incomplete change data from Node {node_id}: {data.hex()}")
            return None

        change_count = data[2]
        parsed_data = {}
        index = 3

        print(f"[DEBUG] Change count: {change_count}")

        for _ in range(change_count):
            if index >= len(data):
                print(f"[ERROR] Incomplete change data for Node {node_id}")
                return None

            pin = data[index]
            index += 1
            print(f"[DEBUG] Parsing pin: {pin}, Index now at: {index}")

            # Detect if it's analog or digital
            if pin < 8:  # Analog pin
                print(f"[DEBUG] Detected analog pin: {pin}")
                if index + 1 >= len(data):
                    print(f"[ERROR] Incomplete analog data for Node {node_id}")
                    return None
                low_byte = data[index]
                high_byte = data[index + 1]
                value = low_byte | (high_byte << 8)
                parsed_data[pin] = value
                index += 2
                print(f"[DEBUG] Analog pin {pin} value: {value}")

            else:  # Digital pin (8 or 9)
                print(f"[DEBUG] Detected digital pin: {pin}")
                if index >= len(data):
                    print(f"[ERROR] Incomplete digital data for Node {node_id}")
                    return None
                value = data[index]
                parsed_data[pin] = value
                index += 1
                print(f"[DEBUG] Digital pin {pin} value: {value}")

        print(f"[DEBUG] Final parsed data for Node {node_id}: {parsed_data}")
        return node_id, parsed_data

    print(f"[ERROR] Unknown flag {flag} from Node {node_id}")
    return None


def poll_single_node(ser, node_id):
    """ Poll a single node with retry logic """
    attempt = 0

    while attempt < MAX_RETRIES:
        print(f"[INFO] Polling Node {node_id}, Attempt {attempt + 1}...")

        # Clear buffer and request data
        ser.reset_input_buffer()
        ser.write(bytes([node_id]))
        print(f"[DEBUG] Request sent to Node {node_id}")

        # Read response
        buffer = bytearray()
        reading = False

        start_time = time.time()
        while time.time() - start_time < TIMEOUT:
            byte = ser.read()
            if not byte:
                continue
            byte = byte[0]

            if byte == START_BYTE:
                buffer = bytearray()
                reading = True
                print(f"[INFO] Start byte detected for Node {node_id}")
            elif byte == END_BYTE and reading:
                print(f"[INFO] End byte detected for Node {node_id}")
                break
            elif reading:
                buffer.append(byte)

        if len(buffer) > 0:
            print(f"[DEBUG] Received data from Node {node_id}: {buffer.hex()}")
            return buffer

        print(f"[WARN] No valid response from Node {node_id}, retrying...")
        attempt += 1

    print(f"[ERROR] Node {node_id} failed after {MAX_RETRIES} retries.")
    return None


def poll_nodes():
    """ Poll all nodes, parse their responses and profile timings """
    node_data = {}
    timings = defaultdict(float)
    missed_nodes = []

    with serial.Serial(PORT, BAUD, timeout=TIMEOUT) as ser:
        time.sleep(1)

        for node_id in range(1, 21):
            t_start = time.time()

            # Poll the node
            buffer = poll_single_node(ser, node_id)
            t_read = time.time()

            # Parse the response
            if buffer is None:
                print(f"[WARN] Missed response from Node {node_id}")
                missed_nodes.append(node_id)
                continue

            parsed = parse_node_response(buffer)
            if parsed:
                node_id, changes = parsed
                node_data[node_id] = changes
            else:
                print(f"[ERROR] Failed to parse response from Node {node_id}: {buffer.hex()}")
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
            print(f"Node {node}: {node_data[node]}")
        else:
            print(f"Node {node}: ❌ Missed")

    if len(missed_nodes) == 0:
        print("\n✅ All nodes successfully parsed!")
    else:
        print(f"\n❌ Missed Nodes: {missed_nodes}")

    return node_data


# === Run Multiple Simulations ===
for run in range(1, RUNS + 1):
    print(f"\n\n=== Simulation Run {run} ===")
    poll_nodes()
