import jax.numpy as jnp

JUMP_DURATION = 0.6  # Total seconds for the full jump cycle
JUMP_AMPLITUDE = 0.4 # Strength of the jump deflection

def jump(t, amplitude=JUMP_AMPLITUDE, duration=JUMP_DURATION):
    # Normalize time between 0 and 1 over the jump duration
    phase = jnp.clip(t / duration, 0.0, 1.0)
    
    # Define phase boundaries
    crouch_end = 0.3
    launch_end = 0.5
    
    # Joint behaviors per phase using smooth interpolation/conditionals
    # Crouch phase: bend joints (negative femur, positive tibia)
    crouch_factor = jnp.clip(phase / crouch_end, 0.0, 1.0)
    
    # Launch phase: rapid extension (positive femur, negative tibia push)
    launch_progress = jnp.clip((phase - crouch_end) / (launch_end - crouch_end), 0.0, 1.0)
    launch_factor = jnp.sin(launch_progress * jnp.pi / 2) # smooth push
    
    # Air/Recovery phase: return smoothly to neutral or tucked
    recovery_progress = jnp.clip((phase - launch_end) / (1.0 - launch_end), 0.0, 1.0)
    
    # Compute joint values based on current active phase
    # Coxa stays mostly neutral during a pure vertical jump
    coxa = jnp.zeros(6)
    
    # Femur trajectory: negative during crouch, high positive push during launch, resetting in air
    femur_crouch = -amplitude * 1.5 * crouch_factor
    femur_launch = amplitude * 2.5 * launch_factor
    femur_recovery = -amplitude * 1.5 * (1.0 - recovery_progress)
    
    femur = jnp.where(
        phase <= crouch_end, 
        femur_crouch, 
        jnp.where(phase <= launch_end, femur_launch, femur_recovery)
    )
    
    # Tibia trajectory mirrors femur to extend/bend knees
    tibia_crouch = amplitude * 1.2 * crouch_factor
    tibia_launch = -amplitude * 2.0 * launch_factor
    tibia_recovery = amplitude * 0.5 * recovery_progress
    
    tibia = jnp.where(
        phase <= crouch_end, 
        tibia_crouch, 
        jnp.where(phase <= launch_end, tibia_launch, tibia_recovery)
    )
    
    # femur/tibia are scalars (all legs move together), so expand them to one per leg
    femur = jnp.broadcast_to(femur, coxa.shape)
    tibia = jnp.broadcast_to(tibia, coxa.shape)

    # Stack along the last axis and flatten to [18]
    return jnp.stack([coxa, femur, tibia], -1).reshape(-1)
