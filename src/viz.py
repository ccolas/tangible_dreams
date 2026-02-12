import moderngl
import numpy as np
from abc import ABC, abstractmethod
import time
import pygame
from pygame.locals import OPENGL, DOUBLEBUF, RESIZABLE, FULLSCREEN
import rtmidi
import asyncio
from src.github_save import save_and_push
import os, sys

try:
    from screeninfo import get_monitors

    monitors = get_monitors()
    laptop = next(m for m in monitors if "eDP" in m.name)
    external = next(m for m in monitors if "HDMI" in m.name)
except Exception as e:
    print(f"Could not detect monitors automatically: {e}")


    # Fallback - define monitor info manually
    class Monitor:
        def __init__(self, x, y, width, height, width_mm, height_mm, name, is_primary):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.name = name
            self.is_primary = is_primary
            self.width_mm = width_mm
            self.height_mm = height_mm

    laptop = Monitor(x=0, y=0, width=1920, height=1080, width_mm=344, height_mm=194, name='eDP-1', is_primary=True)
    external = Monitor(x=1920, y=0, width=1920, height=1200, width_mm=1600, height_mm=900, name='HDMI-1-0', is_primary=False)

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

class VisualizationBackend(ABC):
    @abstractmethod
    def initialize(self, controller, width: int, height: int, window_scale: float = 0.5):
        pass

    @abstractmethod
    def update(self, image_data):
        pass

    @abstractmethod
    def cleanup(self):
        pass



class ModernGLBackend:
    def __init__(self):
        pygame.init()
        self.midi_in = rtmidi.MidiIn()
        self.setup_midi()

    def setup_midi(self):
        ports = self.midi_in.get_ports()
        for i, name in enumerate(ports):
            if 'nanoKONTROL' in name:
                self.midi_in.open_port(i)
                print('[MIDI] connected')
                return
        print("[MIDI] No MIDI device found")


    def initialize(self, cppn, screen_loc, width: int, height: int, window_scale: float):
        self.cppn = cppn
        self.render_width = width
        self.render_height = height

        # postprocessing parameters controlled by MIDI
        self.grain_strength = 0.0
        self.displace_strength = 0.0
        self.chromatic_shift = 0.0
        self.symmetry_mode = 0  # 0,1,2,3,4,...
        self.invert = False
        self.needs_update = False
        self.measured_delay = 0.030  # updated by main loop

        # Create an OpenGL-enabled Pygame window
        flags = DOUBLEBUF | OPENGL
        if screen_loc == 'laptop':
            window_width = laptop.width
            window_height = laptop.height
            flags = flags | FULLSCREEN
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{laptop.x},{laptop.y}"
            os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"  # don't minimize if you click somewhere else
        elif screen_loc == 'external':
            window_width = external.width
            window_height = external.height
            flags = flags | FULLSCREEN
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{external.x},{external.y}"
            os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"    # don't minimize if you click somewhere else
        else:
            window_width = int(width * window_scale)
            window_height = int(height * window_scale)
        self.screen = pygame.display.set_mode((window_width, window_height), flags)

        pygame.display.set_caption("Tangible Dreams")

        # Create ModernGL context from the Pygame window
        self.ctx = moderngl.create_context()

        # Create a texture to hold our image
        self.texture = self.ctx.texture((width, height), components=3)
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Fullscreen quad to display the texture
        vertices = np.array([
            -1, -1,  0.0, 0.0,
            1, -1,  1.0, 0.0,
            -1,  1,  0.0, 1.0,
            1,  1,  1.0, 1.0
        ], dtype='f4')

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec2 in_tex;
                out vec2 v_tex;
                void main() {
                    v_tex = in_tex;
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                
                uniform sampler2D Texture;

                // visual effect controls
                uniform float grain_strength;
                uniform float displace_strength;
                uniform float chroma_shift;
                uniform bool invert_colors;
                uniform int symmetry_mode;   // 0 = off, 1..n = different kaleidoscopes
                
                in vec2 v_tex;
                out vec4 f_color;
                
                // simple hash for grain/displacement
                float rand(vec2 co) {
                    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
                }
                
                vec2 kaleidoscope(vec2 uv, int sectors) {
                    // center
                    vec2 c = vec2(0.5, 0.5);
                    vec2 p = uv - c;
                
                    float a = atan(p.y, p.x);
                    float r = length(p);
                
                    // sector angle
                    float sector = 3.141592 * 2.0 / float(sectors);
                
                    // wrap into sector
                    a = mod(a, sector);
                
                    // reflect every second sector
                    if (a > sector * 0.5)
                        a = sector - a;
                
                    return c + r * vec2(cos(a), sin(a));
                }

                void main()
                {
                    vec2 uv = v_tex;
                
                    // --- symmetry ---
                    if (symmetry_mode == 1) uv = kaleidoscope(uv, 6);
                    else if (symmetry_mode == 2) uv = kaleidoscope(uv, 8);
                    else if (symmetry_mode == 3) uv = kaleidoscope(uv, 12);
                    else if (symmetry_mode == 4) uv = kaleidoscope(uv, 4); 
                                   
                    // --- displacement (fraction of screen) ---
                    if (displace_strength > 0.0001) {
                        float d = displace_strength;

                        float dx = (rand(uv * 200.0) - 0.5) * d;
                        float dy = (rand(uv * 300.0) - 0.5) * d;
                        uv += vec2(dx, dy);
                    }
                    uv = clamp(uv, vec2(0.0), vec2(1.0));

                    vec3 col;
                    col = texture(Texture, uv).rgb;
                                    
                    // --- invert ---
                    if (invert_colors)
                        col = 1.0 - col;
                
                    // --- grain ---
                    if (grain_strength > 0.001) {
                        float g = (rand(uv * 500.0) - 0.5) * grain_strength;
                        col += vec3(g);
                    }
                
                    f_color = vec4(col, 1.0);
                }

            '''
        )
        # add uniforms
        self.prog['grain_strength'] = 0.0
        self.prog['displace_strength'] = 0.0
        self.prog['invert_colors'] = False
        self.prog['symmetry_mode'] = 0

        vbo = self.ctx.buffer(vertices.tobytes())
        vao_content = [
            (vbo, '2f 2f', 'in_vert', 'in_tex')
        ]
        self.vao = self.ctx.vertex_array(self.prog, vao_content)
        self.img_buffer = np.empty((height, width, 3), dtype='u1')

    def update(self, image_data):
        # image_data is already uint8 [0-255] on GPU from JAX
        # Convert directly without CPU roundtrip
        img_np = np.asarray(image_data)  # GPU->CPU copy only
        assert img_np.shape == (self.render_height, self.render_width, 3)
        self.texture.write(img_np.tobytes())

        # Render to screen
        self.ctx.clear(0.0, 0.0, 0.0)
        self.texture.use()
        self.prog['grain_strength'].value = self.grain_strength
        self.prog['displace_strength'].value = self.displace_strength
        self.prog['invert_colors'].value = self.invert
        self.prog['symmetry_mode'].value = self.symmetry_mode
        self.vao.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()

    def _viz_params(self):
        return {
            'grain_strength': self.grain_strength,
            'displace_strength': self.displace_strength,
            'invert': self.invert,
            'symmetry_mode': self.symmetry_mode,
        }

    def poll_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.VIDEORESIZE:
                pygame.display.set_mode(event.size, pygame.SCALED | pygame.RESIZABLE | pygame.DOUBLEBUF)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s and self.cppn:
                asyncio.create_task(save_and_push(self.cppn, viz_params=self._viz_params()))
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n and self.cppn:
                self.cppn.print_connections()
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
            #     pygame.display.toggle_fullscreen()

        if self.midi_in:
            self.needs_update = False
            while True:
                msg = self.midi_in.get_message()
                if not msg:
                    break  # queue empty

                data, _ts = msg
                status, control, value = data[0:3]
                # Restart
                if control == 41 and value == 127:
                    os.execv(sys.executable, ['python'] + sys.argv)

                # Save state
                elif control == 45 and value == 127:
                    asyncio.create_task(save_and_push(self.cppn, viz_params=self._viz_params()))

                # Slider layout (CC 0-7):
                #   0: grain, 1: displacement, 2: delay,
                #   3: attack, 4: release, 5: bass gain, 6: mid gain, 7: treble gain
                # Knob layout (CC 16-23):
                #   17: flux decay, 18: bass lo cut, 19: bass/mid xover,
                #   20: mid/treble xover, 21: bass gate%, 22: mid gate%, 23: treble gate%
                # Button: CC 66 = set delay to measured
                v = value / 127.0

                if control == 0:
                    self.grain_strength = v * 0.7
                    self.needs_update = True
                elif control == 1:
                    self.displace_strength = (v ** 2) * 0.03  # 0-5% of screen width
                    self.needs_update = True

                if self.cppn.audio:
                    audio = self.cppn.audio
                    is_flux = getattr(audio, 'mode', 'simple') == 'flux'

                    if is_flux:
                        # Flux layout: 2=delay, 3=attack, 4=release, 17=flux_decay, 66=set delay
                        if control == 2:
                            audio.delay_seconds = v * 0.150
                        elif control == 3:
                            audio.alpha_attack = v * 0.95
                        elif control == 4:
                            audio.alpha_release = v * 0.95
                        elif control == 66 and value == 127:
                            audio.delay_seconds = self.measured_delay
                            print(f"[Delay] set to measured: {self.measured_delay*1000:.0f}ms")
                        elif control == 17:
                            audio.flux_decay = 0.5 + v * 0.49
                    else:
                        # Simple layout: 2=attack, 3=release, 4=delay, 68=set delay
                        if control == 2:
                            audio.alpha_attack = v * 0.95
                        elif control == 3:
                            audio.alpha_release = v * 0.95
                        elif control == 4:
                            audio.delay_seconds = v * 0.150
                        elif control == 68 and value == 127:
                            audio.delay_seconds = self.measured_delay
                            print(f"[Delay] set to measured: {self.measured_delay*1000:.0f}ms")

                    # Shared controls (both modes)
                    if control == 5:
                        audio.band_gain['bass'] = (v ** 2) * 8.0
                    elif control == 6:
                        audio.band_gain['mid'] = (v ** 2) * 8.0
                    elif control == 7:
                        audio.band_gain['treble'] = (v ** 2) * 8.0
                    elif control == 18:
                        freq = 20.0 + v * 80.0
                        audio.bands['bass'] = (freq, audio.bands['bass'][1])
                        audio.update_with_bands()
                    elif control == 19:
                        freq = 50.0 + v * 450.0
                        audio.bands['bass'] = (audio.bands['bass'][0], freq)
                        audio.bands['mid'] = (freq, audio.bands['mid'][1])
                        audio.update_with_bands()
                    elif control == 20:
                        freq = 1000.0 + v * 5000.0
                        audio.bands['mid'] = (audio.bands['mid'][0], freq)
                        audio.bands['treble'] = (freq, audio.bands['treble'][1])
                        audio.update_with_bands()
                    elif control == 21:
                        audio.gate_target_fraction['bass'] = v * 0.40
                    elif control == 22:
                        audio.gate_target_fraction['mid'] = v * 0.40
                    elif control == 23:
                        audio.gate_target_fraction['treble'] = v * 0.40

                if control == 60 and value == 127:
                    self.invert = not self.invert
                    self.needs_update = True
                if control == 46 and value == 127:
                    self.symmetry_mode = (self.symmetry_mode + 1) % 6
                    self.needs_update = True

                # audio band control
                # visual control



    def cleanup(self):
        pygame.quit()




class PygameBackend(VisualizationBackend):
    def __init__(self):
        pygame.init()

    def initialize(self, cppn, width: int, height: int, window_scale: float = 0.5):
        self.cppn = cppn
        self.render_w, self.render_h = int(width), int(height)

        flags = pygame.SCALED | pygame.RESIZABLE | pygame.DOUBLEBUF | pygame.FULLSCREEN
        try:
            self.screen = pygame.display.set_mode((self.render_w, self.render_h), flags, vsync=0)
        except TypeError:
            self.screen = pygame.display.set_mode((self.render_w, self.render_h), flags)

        self.surface = pygame.Surface((self.render_w, self.render_h)).convert(self.screen)
        pygame.event.set_allowed([pygame.QUIT, pygame.VIDEORESIZE, pygame.KEYDOWN])

    def update(self, image_data):
        # image_data: HxWx3 uint8
        frame = np.asarray(image_data, dtype=np.uint8, order="C")

        # Pygame surfarray expects (W,H,3); use a view, no copy
        whc = np.swapaxes(frame, 0, 1)
        pygame.surfarray.blit_array(self.surface, whc)
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            elif e.type == pygame.VIDEORESIZE:
                # SCALED handles the stretching; just update the window size
                pygame.display.set_mode(e.size, pygame.SCALED | pygame.RESIZABLE | pygame.DOUBLEBUF)
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_s and self.cppn:
                import asyncio
                from src.github_save import save_and_push
                asyncio.create_task(save_and_push(self.cppn))
        return True

    def cleanup(self):
        pygame.quit()

def create_backend(backend_type='moderngl'):
    if backend_type == 'moderngl':
        return ModernGLBackend()
    elif backend_type == 'pygame':
        return PygameBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

