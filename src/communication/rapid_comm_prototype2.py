import serial
import time
import os
import signal
import atexit
from collections import defaultdict

# === CONFIG ===
PORT = '/dev/ttyUSB0'
BAUD = 115200
START_BYTE = 0xAA
END_BYTE = 0xBB
TIMEOUT = 0.004  # 4ms for update
TIMEOUT_TOTAL = 1  # generous first/full read
MAX_RETRIES = 3
NODES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
NODES_TO_PLOT = NODES

# === Latency check ===
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

# === Open / Close with bus park ===
def open_port():
    ser = serial.Serial(
        PORT, BAUD,
        timeout=0, write_timeout=0,
        rtscts=False, dsrdtr=False, xonxoff=False
    )
    # drain junk
    ser.reset_input_buffer(); ser.reset_output_buffer()
    t0 = time.time()
    while time.time() - t0 < 0.05:
        ser.read(4096)

    # idle preamble so nodes resync
    ser.write(b'\xFF' * 64)
    ser.flush()
    time.sleep(0.003)

    try:
        ser.setDTR(False); ser.setRTS(False)
    except Exception:
        pass
    return ser

def park_and_close(ser):
    try:
        ser.write(b'\xFF' * 64)
        ser.flush()
        time.sleep(0.003)
        ser.reset_input_buffer(); ser.reset_output_buffer()
        try:
            ser.setDTR(False); ser.setRTS(False)
        except Exception:
            pass
    except Exception:
        pass
    try:
        ser.close()
    except Exception:
        pass

# ensure cleanup on exit / ctrl-c
def install_clean_exit(ser):
    atexit.register(lambda: park_and_close(ser))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(SystemExit))

# === Parsing ===
def parse_node_response(data):
    if len(data) < 2:
        return None
    node_id = data[0]
    flag = data[1]
    if flag == 0x01:
        return node_id, {}
    if flag == 0xFF:
        if len(data) < 3:
            return None
        change_count = data[2]
        parsed_data = {}
        index = 3
        for _ in range(change_count):
            if index >= len(data):
                return None
            pin = data[index]; index += 1
            if pin < 8:
                if index + 1 >= len(data):
                    return None
                value = data[index] | (data[index + 1] << 8)
                index += 2
            elif pin < 10:
                if index >= len(data):
                    return None
                value = data[index]; index += 1
            else:
                return None
            parsed_data[pin] = value
        return node_id, parsed_data
    return None

# === Polling ===
def poll_single_node(ser, node_id, command, timeout):
    ser.reset_input_buffer()
    ser.write(bytes([0xCC, node_id, command]))
    ser.flush()
    time.sleep(0.0004)  # guard ~200µs

    buffer = bytearray()
    reading = False
    start_time = time.time()

    while time.time() - start_time < timeout:
        byte = ser.read()
        if not byte:
            continue
        byte = byte[0]
        if byte == START_BYTE:
            buffer = bytearray(); reading = True
        elif byte == END_BYTE and reading:
            break
        elif reading:
            buffer.append(byte)

    if len(buffer) >= 2:
        return buffer
    return None

def poll_nodes():
    parsed_once = False
    ser = open_port()
    install_clean_exit(ser)

    time.sleep(0.1)  # let bus settle
    run = 0
    while True:
        run += 1
        node_data = {}
        missed_nodes = []
        timings = defaultdict(float)

        command = 0x01 if (run == 1 or not parsed_once) else 0x00
        timeout = TIMEOUT_TOTAL if command == 0x01 else TIMEOUT
        max_retries = 10 if command == 0x01 else MAX_RETRIES

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
                    if parsed:
                        break
                if command == 0x01 and timeout < 1.0:
                    timeout += 0.05  # adaptive first-pass
            if parsed:
                node_data[node_id] = parsed[1]
                parsed_once = True
            else:
                missed_nodes.append(node_id)
                print(f"[FAIL] Node {node_id} (after {max_retries} attempts)")
            i_attempts.append(i_attempt)

        any_change = False
        for node_id, i_attempt in zip(NODES, i_attempts):
            if node_id in node_data:
                if node_id in NODES_TO_PLOT and len(node_data[node_id]) > 0:
                    any_change = True
                    print(f"  Node {node_id}: {node_data[node_id]}; i_attempts={i_attempt}")
            else:
                print(f"  Node {node_id}: ❌")

        if any_change:
            total_time = sum(timings.values())
            print(f"[Timing] Total: {total_time:.2f} ms")

if __name__ == "__main__":
    poll_nodes()
