"""Move any knob/slider/button on your MIDI controller to see its CC number and value."""

import rtmidi

midi_in = rtmidi.MidiIn()
ports = midi_in.get_ports()

if not ports:
    print("No MIDI devices found.")
    exit(1)

print("Available MIDI ports:")
PORT = None
for i, name in enumerate(ports):
    print(f"  [{i}] {name}")
    if "nanoKONTROL" in name:
        PORT = i
        break

# Open first port (change index if needed)
midi_in.open_port(PORT)
print(f"\nListening on: {ports[PORT]}")
print("Move controls to see CC values. Ctrl+C to quit.\n")

try:
    while True:
        msg = midi_in.get_message()
        if msg:
            data, timestamp = msg
            if len(data) == 3:
                status, control, value = data
                ch = (status & 0x0F) + 1
                msg_type = (status & 0xF0)
                if msg_type == 0xB0:
                    print(f"CC {control:>3d}  value={value:>3d}  (ch {ch})")
                elif msg_type == 0x90:
                    print(f"NOTE ON  note={control:>3d}  velocity={value:>3d}  (ch {ch})")
                elif msg_type == 0x80:
                    print(f"NOTE OFF note={control:>3d}  (ch {ch})")
                else:
                    print(f"status=0x{status:02X}  d1={control}  d2={value}")
except KeyboardInterrupt:
    print("\nDone.")
finally:
    midi_in.close_port()