import moderngl
import numpy as np
# import cv2
import jax.numpy as jnp
from abc import ABC, abstractmethod
import time
import pygame
import asyncio
from src.github_save import save_and_push
from pygame.locals import OPENGL, DOUBLEBUF, RESIZABLE
import jax


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

    def initialize(self, cppn, width: int, height: int, window_scale: float):
        self.cppn = cppn
        self.render_width = width
        self.render_height = height

        # Window size vs render resolution
        window_width = int(width * window_scale)
        window_height = int(height * window_scale)

        # Create an OpenGL-enabled Pygame window
        self.screen = pygame.display.set_mode((window_width, window_height), DOUBLEBUF | OPENGL)
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
                in vec2 v_tex;
                out vec4 f_color;
                void main() {
                    f_color = texture(Texture, v_tex);
                }
            '''
        )

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
        self.texture.write(img_np.tobytes())

        # Render to screen
        self.ctx.clear(0.0, 0.0, 0.0)
        self.texture.use()
        self.vao.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()

    def poll_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.VIDEORESIZE:
                pygame.display.set_mode(event.size, pygame.SCALED | pygame.RESIZABLE | pygame.DOUBLEBUF)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s and self.cppn:
                import asyncio
                from src.github_save import save_and_push
                asyncio.create_task(save_and_push(self.cppn))
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n and self.cppn:
                self.cppn.print_connections()


    def cleanup(self):
        pygame.quit()




class PygameBackend(VisualizationBackend):
    def __init__(self):
        pygame.init()

    def initialize(self, cppn, width: int, height: int, window_scale: float = 0.5):
        self.cppn = cppn
        self.render_w, self.render_h = int(width), int(height)

        flags = pygame.SCALED | pygame.RESIZABLE | pygame.DOUBLEBUF
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

