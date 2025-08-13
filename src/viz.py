# import moderngl
import numpy as np
# import cv2
import jax.numpy as jnp
from abc import ABC, abstractmethod
import time
import pygame
import asyncio
from src.github_save import save_and_push


class VisualizationBackend(ABC):
    """Abstract base class for visualization backends"""

    @abstractmethod
    def initialize(self, controller, width: int, height: int):
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

    def initialize(self, controller, render_width: int, render_height: int,
                   window_scale: float = 0.4):  # Scale factor for window size
        # Keep track of full resolution for rendering
        self.controller = controller
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
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s and self.controller:
                    print('SAVING state and image and pushing to repo')
                    asyncio.create_task(save_and_push(self.controller.cppn, self.controller.output_path))

        times = {
            'permute': t1 - t0,
            'convert': t2 - t1,
            'display': t3 - t2,
            'total': t3 - t0
        }
        return times

    def cleanup(self):
        pygame.quit()



def create_backend(backend_type='moderngl'):
    """Factory function to create visualization backend"""
    if backend_type == 'pygame':
        return PygameBackend()
    # elif backend_type == 'moderngl':
    #     return ModernGLBackend()
    # elif backend_type == 'opencv':
    #     return OpenCVBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")