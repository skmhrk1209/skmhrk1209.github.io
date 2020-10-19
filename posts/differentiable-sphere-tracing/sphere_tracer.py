import torch
from torch import nn
from torch import autograd


class SphereTracer(autograd.Function):

    @staticmethod
    def forward(ctx, sdf, sdf_parameters, latents, cameras, num_marching_iterations, convergence_criteria):

        y_coordinates = torch.arange(latents.shape[-2], dtype=latents.dtype, device=latents.device)
        x_coordinates = torch.arange(latents.shape[-1], dtype=latents.dtype, device=latents.device)
        y_coordinates, x_coordinates = torch.meshgrid(y_coordinates, x_coordinates)
        z_coordinates = torch.ones_like(y_coordinates)
        pixel_coordinates = torch.stack((x_coordinates, y_coordinates, z_coordinates), dim=0).unsqueeze(0)
        world_coordinates = cameras.unproject_points(pixel_coordinates, world_coordinates=True)
        ray_directions = nn.functional.normalize(world_coordinates - cameras.get_camera_center(), dim=1)

        # sphere tracing algorithm
        with torch.no_grad():
            ray_positions = world_coordinates
            for i in range(num_marching_iterations):
                sdf_values = sdf(latents, ray_positions)
                ray_positions = ray_directions * sdf_values
                if torch.all(torch.abs(sdf_values) < convergence_criteria):
                    break
        
        # save values for backward pass
        ctx.sdf = sdf
        ctx.sdf_parameters = sdf_parameters
        ctx.save_for_backward(latents, ray_directions, ray_positions)

    @staticmethod
    def backward(ctx, grad_outputs):
        
        latents, ray_directions, ray_positions = ctx.saved_tensors
        sdf_parameters = ctx.sdf_parameters
        sdf = ctx.sdf

        # compute gradients using implicit differentiation
        with torch.enable_grad():

            sdf_values = sdf(latents, ray_positions)
            grad_positions, = autograd.grad(sdf_values, ray_positions, grad_outputs=torch.ones_like(sdf_values), retain_graph=True)
            # grad_positions = map(torch.stack, autograd.grad(torch.unbind(sdf_values), [ray_positions] * sdf_values.shape[0], retain_graph=True))
            grad_positions_dot_directions = torch.sum(grad_positions * ray_directions, dim=1, keepdim=True)
            grad_outputs = -grad_outputs / grad_positions_dot_directions

            grad_sdf_parameters = autograd.grad(sdf_values, sdf_parameters, grad_outputs=grad_outputs, retain_graph=True)
            grad_latents, = torch.autograd.grad(sdf_values, latents, grad_outputs=grad_outputs, retain_graph=True)

        return None, grad_sdf_parameters, grad_latents, None, None, None
