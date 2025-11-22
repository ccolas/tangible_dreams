import serial, time

PORT = '/dev/ttyUSB0'
BAUD = 115200          # try 57600 if needed
STUCK = [16]     # nodes to revive

def open_and_wake():
    ser = serial.Serial(PORT, BAUD, timeout=0, write_timeout=0)
    # settle and drain
    ser.reset_input_buffer(); ser.reset_output_buffer()
    time.sleep(0.05)
    t0 = time.time()
    while time.time() - t0 < 0.05:
        ser.read(4096)

    # idle preamble (mark) so receivers re-lock
    ser.write(b'\xFF' * 64)
    ser.flush()
    time.sleep(0.003)
    return ser

def read_frame(ser, timeout):
    start = time.time()
    reading = False
    buf = bytearray()
    while time.time() - start < timeout:
        b = ser.read()
        if not b:
            continue
        b = b[0]
        if b == 0xAA:           # START
            buf.clear(); reading = True
        elif b == 0xBB and reading:  # END
            return bytes(buf)
        elif reading:
            buf.append(b)
    return None

def ping_node(ser, node, first_read=True):
    cmd = 0x01 if first_read else 0x00
    # small idle gap before TX
    time.sleep(0.001)
    ser.write(bytes([0xCC, node, cmd]))
    ser.flush()
    # line turnaround guard
    time.sleep(0.0003)
    frame = read_frame(ser, 0.50 if first_read else 0.02)
    if frame:
        # frame = [node_id, flag, ...]
        print(f"Node {node} → {frame.hex()}")
        return True
    print(f"Node {node} ❌")
    return False

if __name__ == "__main__":
    ser = open_and_wake()
    # try a few times with first-read semantics
    for _ in range(5):
        ok = True
        for n in STUCK:
            ok &= ping_node(ser, n, first_read=True)
        if ok: break
    ser.close()
