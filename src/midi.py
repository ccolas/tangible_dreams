import rtmidi
import asyncio

from src.midi_cppn import CPPN


class MIDIController:
    def __init__(self, output_path, params):
        self.midi_in = rtmidi.MidiIn()
        self.output_path = output_path
        self.cppn = CPPN(output_path, params)
        self.params = params
        self.debug = params['debug']
        self.setup_midi()

    def setup_midi(self):
        ports = self.midi_in.get_ports()
        midi_connected = False
        for i, name in enumerate(ports):
            if 'nanoKONTROL' in name:
                self.midi_in.open_port(i)
                print('Midi connected')
                midi_connected = True
                break
        assert midi_connected
        if not self.debug:
            self.midi_in.set_callback(self.midi_callback)

    async def start_polling_loop(self):
        while True:
            self.check_midi()
            await asyncio.sleep(0.001)  # ~1000Hz polling rate

    def check_midi(self):
        # Poll and process all pending MIDI messages this frame (max 100 to avoid lockups)
        for _ in range(100):
            msg = self.midi_in.get_message()
            if not msg:
                break
            self.midi_callback(msg, None)

    def midi_callback(self, event, _):
        msg, _ = event
        if msg[0] in [176, 144]:  # Knobs and buttons
            control = msg[1]
            if msg[0] == 176:
                control_type = 'knob'
            else:
                control_type = 'button'
        else:  # Sliders
            control_type = 'slider'
            control = msg[0]
        value = msg[2]
        print(control_type, control, value, self.cppn.needs_update)

        # Handle SET button
        if control == 82:
            self.cppn.set_pressed = (value == 127)
            return

        if self.cppn.set_pressed:
            # Alternative controls when SET is pressed
            if control == 224:  # First slider for zoom
                value = 0.01 * (10000) ** (value / 127)  # use exponential mapping
                self.cppn.zoom = value
                self.cppn.inputs[self.cppn.res] = self.cppn.generate_inputs(self.cppn.res)
                self.cppn.needs_update = True
                print(f'Setting zoom to {value}')
            elif 225 <= control <= 226:  # First 2 knobs for panning
                value = 3 * (value / 64 - 1)  # Map to [-3, 3]
                if control == 225:
                    self.cppn.x_offset = value
                    print(f'Setting x offset to {value}')
                elif control == 226:
                    self.cppn.y_offset = value
                    print(f'Setting y offset to {value}')
                self.cppn.inputs[self.cppn.res] = self.cppn.generate_inputs(self.cppn.res)
                self.cppn.needs_update = True
            elif 229 <= control <= 231:  # Last three sliders for RGB slopes
                slope_idx = control - 229
                slope = 0.1 * (100) ** (value / 127)
                self.cppn.rgb_slopes = self.cppn.rgb_slopes.at[slope_idx].set(slope)
                self.cppn.needs_update = True
                print(f'Setting {["red", "green", "blue"][slope_idx]} tanh slope to {slope}')
            elif control_type == 'knob' and 21 <= control <= 23:  # Last three knobs for RGB biases
                channel = control - 21
                if value == 2:  # Right turn
                    new_value = self.cppn.rgb_biases[channel] + 0.1
                elif value == 66:  # Left turn
                    new_value = self.cppn.rgb_biases[channel] - 0.1
                else:
                    return

                self.cppn.rgb_biases = self.cppn.rgb_biases.at[channel].set(new_value)
                self.cppn.needs_update = True
                print(f'RGB bias {channel}: {new_value:.2f}')
        else:
            if control == 94 and value == 127:  # Play - sample new network
                self.cppn.sample_network()
                self.cppn.inputs[self.cppn.res] = self.cppn.generate_inputs(self.cppn.res)
                self.cppn.needs_update = True
            elif control == 95 and value == 127:  # Record - save state
                self.cppn.save_state()
            elif control == 91 and value == 127:  # Previous - major history
                self.cppn.major_index = max(0, self.cppn.major_index - 1)
                print(f'loading major state {self.cppn.major_index}')
                self.cppn.load_major_state()
            elif control == 92 and value == 127:  # Next - major history
                self.cppn.major_index = min(len(self.cppn.major_history) - 1, self.cppn.major_index + 1)
                print(f'loading major state {self.cppn.major_index}')
                self.cppn.load_major_state()
            elif control == 84 and value == 127:  # Left - minor history
                self.cppn.minor_index = max(0, self.cppn.minor_index - 1)
                print(f'loading minor state {self.cppn.minor_index}')
                self.cppn.load_minor_state()
            elif control == 85 and value == 127:  # Right - minor history
                self.cppn.minor_index = min(len(self.cppn.minor_history) - 1, self.cppn.minor_index + 1)
                print(f'loading minor state {self.cppn.minor_index}')
                self.cppn.load_minor_state()
            elif control_type == 'button' and 8 <= control <= 15 and value == 127:  # S buttons - change activation
                node_id = control - 8
                self.cppn.activation_ids = self.cppn.activation_ids.at[node_id].set(
                    (self.cppn.activation_ids[node_id] + 1) % len(self.cppn.activations)
                )
                self.cppn.save_minor_state()
                self.cppn.needs_update = True
                print(f'New activation {node_id}: {self.cppn.activations[self.cppn.activation_ids[node_id]]}')
            elif control_type == 'button' and 16 <= control <= 23 and value == 127:  # M buttons - resample output connections
                node_id = control - 16
                self.cppn.resample_out_connections(node_id)
                print(f'Resampled output connections for node {node_id}')
            elif control_type == 'button' and 0 <= control <= 7 and value == 127:  # R buttons - resample input connections
                node_id = control
                self.cppn.resample_in_connections(node_id)
                print(f'Resampled input connections for node {node_id}')
            elif 224 <= control <= 231:  # Sliders - bias
                node_id = control - 224
                bias = 2 * (value / 64 - 1)  # Map to [-5, 5]
                self.cppn.biases = self.cppn.biases.at[node_id].set(bias)
                self.cppn.needs_update = True
                print(f'Bias {node_id}: {bias:.2f}')
            elif control_type == 'knob' and 16 <= control <= 23:  # Knobs - weight multiplier
                node_id = control - 16
                if value == 2:  # Turning right
                    increment = 0.03  # Small increment for smooth changes
                    new_value = self.cppn.multipliers[node_id] + increment
                elif value == 66:  # Turning left
                    increment = 0.03
                    new_value = self.cppn.multipliers[node_id] - increment
                else:
                    new_value = self.cppn.multipliers[node_id]
                self.cppn.multipliers = self.cppn.multipliers.at[node_id].set(new_value)
                self.cppn.needs_update = True
                print(f'Weight {node_id}: {new_value:.2f}')

