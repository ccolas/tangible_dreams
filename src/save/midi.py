import rtmidi
import asyncio

from src.cppn import CPPN, zoom_mapping, bias_mapping


class MIDIController:
    def __init__(self, output_path, params):
        self.midi_in = rtmidi.MidiIn()
        self.output_path = output_path
        params['n_middle_nodes'] = 8
        self.cppn = CPPN(output_path, params)
        self.params = params
        self.debug = params['debug']
        self.setup_midi()
        self.set_pressed = False

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
        self.cppn.reactive_update()
        # self.cppn.update_cam()

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
            self.set_pressed = (value == 127)
            return

        if self.set_pressed:
            # Alternative controls when SET is pressed
            if control == 224:  # First slider for zoom
                value = value / 127 * 1023
                # value = 0.01 * (10000) ** (value / 127)  # use exponential mapping
                self.cppn.input_params1 = self.cppn.input_params1.at[0].set(value)
                self.cppn.input_params1 = self.cppn.input_params1.at[1].set(value)
                # self.cppn.input_params1 = self.cppn.input_params1.at[2].set(value)
                # self.cppn.input_params1 = self.cppn.input_params1.at[1].set(value)
                print(f'Setting zoom to {value}')
            elif 225 <= control <= 226:  # First 2 knobs for panning
                # value = 3 * (value / 64 - 1)  # Map to [-3, 3]
                value = value / 127 * 1023
                if control == 225:
                    self.cppn.input_params2 = self.cppn.input_params2.at[0].set(value)
                    print(f'Setting x offset to {value}')
                elif control == 226:
                    self.cppn.input_params2 = self.cppn.input_params2.at[1].set(value)
                    print(f'Setting y offset to {value}')
            elif control == 227:
                value = value / 127 * 1023
                # value = 0.01 * (10000) ** (value / 127)  # use exponential mapping
                self.cppn.input_params1 = self.cppn.input_params1.at[2].set(value)
                print(f'Setting input 2 param 1 to {value}')
            elif control == 228:
                value = value / 127 * 1023
                # value = 3 * (value / 64 - 1)  # Map to [-3, 3]
                self.cppn.input_params2 = self.cppn.input_params2.at[2].set(value)
                print(f'Setting input 2 param 2 to {value}')

            elif 229 <= control <= 231:  # Last three sliders for RGB slopes
                slope_idx = control - 229
                slope = 0.1 * (100) ** (value / 127)
                self.cppn.output_slopes = self.cppn.output_slopes.at[slope_idx].set(slope)
                self.cppn.needs_update = True
                print(f'Setting {["red", "green", "blue"][slope_idx]} tanh slope to {slope}')
            elif control_type == 'knob' and 21 <= control <= 23:  # Last three knobs for RGB biases
                channel = control - 21
                if value == 2:  # Right turn
                    new_value = self.cppn.output_biases[channel] + 0.1
                elif value == 66:  # Left turn
                    new_value = self.cppn.output_biases[channel] - 0.1
                else:
                    return

                self.cppn.output_biases = self.cppn.output_biases.at[channel].set(new_value)
                self.cppn.needs_update = True
                print(f'RGB bias {channel}: {new_value:.2f}')
        else:
            if control == 94 and value == 127:  # Play - sample new network
                self.cppn.sample_network()
            elif control == 95 and value == 127:  # Record - save state
                self.cppn.save_state()

            elif control_type == 'button' and 8 <= control <= 15 and value == 127:
                hidden_index = control - 8
                if hidden_index < self.cppn.n_hidden:
                    current_id = int(self.cppn.activation_ids[hidden_index])
                    new_activ_id = (current_id + 1) % len(self.cppn.activations)
                    self.cppn.activation_ids = self.cppn.activation_ids.at[hidden_index].set(new_activ_id)
                    print(f'Hidden node {hidden_index} activation: {new_activ_id} ({self.cppn.activations[new_activ_id].__name__})')
                    self.cppn.needs_update = True

            # elif control_type == 'button' and 16 <= control <= 23 and value == 127:  # M buttons - resample output connections
            #     node_id = control - 16
            #     self.cppn.resample_out_connections(node_id)
            #     print(f'Resampled output connections for node {node_id}')
            # elif control_type == 'button' and 0 <= control <= 7 and value == 127:  # R buttons - resample input connections
            #     node_id = control
            #     self.cppn.resample_in_connections(node_id)
            #     print(f'Resampled input connections for node {node_id}')
            elif 224 <= control <= 231:  # Sliders → biases
                hidden_index = control - 224
                if hidden_index < self.cppn.n_hidden:
                    bias = 2 * (value / 64 - 1)  # Map to [-2, 2]
                    self.cppn.biases = self.cppn.biases.at[hidden_index].set(bias)
                    print(f'Hidden node {hidden_index} bias: {bias:.2f}')
                    self.cppn.needs_update = True

            elif control_type == 'knob' and 16 <= control <= 23:  # Knobs → slopes
                hidden_index = control - 16
                if hidden_index < self.cppn.n_hidden:
                    current = float(self.cppn.slopes[hidden_index])
                    if value == 2:
                        new_value = current + 0.03
                    elif value == 66:
                        new_value = current - 0.03
                    else:
                        new_value = current
                    self.cppn.slopes = self.cppn.slopes.at[hidden_index].set(new_value)
                    print(f'Hidden node {hidden_index} slope: {new_value:.2f}')
                    self.cppn.needs_update = True

        self.cppn.needs_update = True

