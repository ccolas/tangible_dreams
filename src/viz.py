import moderngl
import numpy as np
import cv2
import jax.numpy as jnp
from abc import ABC, abstractmethod
import time
import pygame


class VisualizationBackend(ABC):
    """Abstract base class for visualization backends"""

    @abstractmethod
    def initialize(self, width: int, height: int):
        """Initialize the visualization window"""
        pass

    @abstractmethod
    def update(self, image_data):
        """Update the display with new image data"""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources"""
        pass


class PygameBackend(VisualizationBackend):
    def __init__(self):
        pygame.init()

    def initialize(self, render_width: int, render_height: int,
                   window_scale: float = 0.4):  # Scale factor for window size
        # Keep track of full resolution for rendering
        self.render_width = render_width
        self.render_height = render_height

        # Calculate window size
        self.width = int(render_width * window_scale)
        self.height = int(render_height * window_scale)

        # Create window
        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CPPN")

        # Full resolution surface for rendering
        self.surface = pygame.Surface((render_width, render_height))

    def update(self, image_data):
        t0 = time.time()

        # Try to do all conversions while still in JAX
        display_image = np.transpose(np.array((image_data * 255).astype(jnp.uint8)),(1, 0, 2))
        t1 = time.time()

        # Alternative: try pre-allocated numpy buffer
        # if not hasattr(self, 'np_buffer'):
        #     self.np_buffer = np.empty(display_image.shape, dtype=np.uint8)
        # np.multiply(image_data, 255, out=self.np_buffer)

        # Update surface using a temporary array view
        surface_array = pygame.surfarray.pixels3d(self.surface)
        np.copyto(surface_array, display_image)
        del surface_array

        t2 = time.time()

        # Scale to window size and blit
        scaled = pygame.transform.scale(self.surface, (self.width, self.height))
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()

        t3 = time.time()

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.size

        times = {
            'permute': t1 - t0,
            'convert': t2 - t1,
            'display': t3 - t2,
            'total': t3 - t0
        }
        return times

    def cleanup(self):
        pygame.quit()

class ModernGLBackend(VisualizationBackend):
    def __init__(self):
        self.ctx = None
        self.prog = None
        self.vao = None
        self.texture = None

    def initialize(self, width: int, height: int):
        self.width = width
        self.height = height

        # Create OpenGL context
        self.ctx = moderngl.create_context()

        # Create shader program
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec2 in_text;
                out vec2 v_text;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_text = in_text;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D Texture;
                in vec2 v_text;
                out vec4 f_color;
                void main() {
                    f_color = texture(Texture, v_text);
                }
            ''',
        )

        # Create fullscreen quad
        vertices = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype='f4')
        texcoords = np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.tbo = self.ctx.buffer(texcoords.tobytes())

        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '2f', 'in_vert'),
                (self.tbo, '2f', 'in_text'),
            ],
        )

    def update(self, image_data):
        t0 = time.time()

        # Convert to uint8 while still in JAX
        image_data = (image_data * 255).astype(jnp.uint8)

        # Create and update texture
        if self.texture is None:
            self.texture = self.ctx.texture(
                image_data.shape[1::-1],  # width, height
                3,  # components
                image_data.tobytes()
            )
        else:
            self.texture.write(image_data.tobytes())

        t1 = time.time()

        # Render
        self.texture.use()
        self.ctx.clear()
        self.vao.render(moderngl.TRIANGLE_FAN)
        self.ctx.finish()

        t2 = time.time()

        times = {
            'convert': t1 - t0,
            'render': t2 - t1,
            'total': t2 - t0
        }
        return times

    def cleanup(self):
        if self.texture:
            self.texture.release()
        if self.vao:
            self.vao.release()
        if self.prog:
            self.prog.release()
        if self.ctx:
            self.ctx.release()


class OpenCVBackend(VisualizationBackend):
    def __init__(self):
        self.window_name = 'CPPN'

    def initialize(self, width: int, height: int):
        self.width = width
        self.height = height
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width, height)

    def update(self, image_data):
        t0 = time.time()

        # Convert to uint8 while still in JAX
        image_data = (image_data * 255).astype(jnp.uint8)
        # Convert to numpy
        display_image = np.array(image_data)

        t1 = time.time()

        # Convert color space
        display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

        t2 = time.time()

        # Display
        cv2.imshow(self.window_name, display_image)
        cv2.waitKey(1)

        t3 = time.time()

        times = {
            'convert': t1 - t0,
            'color': t2 - t1,
            'display': t3 - t2,
            'total': t3 - t0
        }
        return times

    def cleanup(self):
        cv2.destroyAllWindows()


def create_backend(backend_type='moderngl'):
    """Factory function to create visualization backend"""
    if backend_type == 'moderngl':
        return ModernGLBackend()
    elif backend_type == 'pygame':
        return PygameBackend()
    elif backend_type == 'opencv':
        return OpenCVBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")