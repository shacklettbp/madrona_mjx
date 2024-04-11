from madrona_mjx_sim import SimManager, madrona
import math

class Simulator:
    def __init__(self, gpu_id, num_worlds, cpu_madrona, viz_gpu_hdls=None):
        # Initialize madrona simulator
        
        self.madrona = SimManager(
            exec_mode = madrona.ExecMode.CPU if cpu_madrona else madrona.ExecMode.CUDA,
            gpu_id = gpu_id,
            num_worlds = num_worlds,
            max_episode_length = 500,
            enable_batch_renderer = True,
            batch_render_view_width = 64,
            batch_render_view_height = 64,
            visualizer_gpu_handles = viz_gpu_hdls,
        )
        self.madrona.init()
        
        self.depth = self.madrona.depth_tensor().to_torch()
        self.rgb = self.madrona.rgb_tensor().to_torch()

        self.step_idx = 0

    def step(self):
        self.madrona.process_actions()

        self.madrona.post_physics()

        self.step_idx += 1
