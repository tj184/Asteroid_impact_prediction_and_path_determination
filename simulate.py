import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import rebound
from astropy import constants as const
from astropy import units as u

def initialize_simulation():
    sim = rebound.Simulation()
    sim.add(m=1.0)
    planets = [
        {'name': 'Mercury', 'm': 1.660e-7, 'a': 0.387, 'e': 0.205, 'inc': 7.0, 'Omega': 48.0, 'omega': 29.0, 'M': 0.0},
        {'name': 'Venus', 'm': 2.447e-6, 'a': 0.723, 'e': 0.007, 'inc': 3.4, 'Omega': 77.0, 'omega': 54.0, 'M': 0.0},
        {'name': 'Earth', 'm': 3.003e-6, 'a': 1.0, 'e': 0.0167, 'inc': 0.00005, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0},
        {'name': 'Mars', 'm': 3.227e-7, 'a': 1.5237, 'e': 0.0934, 'inc': 0.032, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0},
        {'name': 'Jupiter', 'm': 9.539e-4, 'a': 5.2044, 'e': 0.0489, 'inc': 0.022, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0},
        {'name': 'Saturn', 'm': 2.858e-4, 'a': 9.582, 'e': 0.0565, 'inc': 0.043, 'Omega': 113.0, 'omega': 92.0, 'M': 0.0},
        {'name': 'Uranus', 'm': 4.366e-5, 'a': 19.191, 'e': 0.046, 'inc': 0.030, 'Omega': 74.0, 'omega': 170.0, 'M': 0.0},
        {'name': 'Neptune', 'm': 5.151e-5, 'a': 30.071, 'e': 0.0086, 'inc': 1.77, 'Omega': 131.0, 'omega': 44.0, 'M': 0.0},
        {'name': 'Pluto', 'm': 7.246e-9, 'a': 39.482, 'e': 0.2488, 'inc': 17.2, 'Omega': 110.0, 'omega': 224.0, 'M': 0.0}
    ]
    for planet in planets:
        try:
            planet_obj = sim.add(m=planet['m'], a=planet['a'], e=planet['e'], inc=planet['inc'], Omega=planet['Omega'], omega=planet['omega'], M=planet['M'])
            planet_obj.label = planet['name']
        except Exception as e:
            print(f"Warning: Planet {planet['name']} could not be added. Error: {e}")
    return sim

def simulate_asteroid_path(asteroid_pos, asteroid_vel, sim, duration=1000):
    sim.add(m=0.0, x=asteroid_pos[0], y=asteroid_pos[1], z=asteroid_pos[2], vx=asteroid_vel[0], vy=asteroid_vel[1], vz=asteroid_vel[2])
    sim.integrator = "ias15"
    sim.dt = 1
    x_asteroid = []
    y_asteroid = []
    z_asteroid = []
    for _ in range(duration):
        sim.integrate(sim.t + sim.dt)
        asteroid = sim.particles[-1]
        x_asteroid.append(asteroid.x)
        y_asteroid.append(asteroid.y)
        z_asteroid.append(asteroid.z)
    return np.array(x_asteroid), np.array(y_asteroid), np.array(z_asteroid)

def animate_simulation(x_asteroid, y_asteroid, z_asteroid, sim, filename="asteroid_simulation.mp4"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, c='yellow', s=300, label="Sun")
    for particle in sim.particles[1:]:
        if hasattr(particle, 'label'):
            ax.scatter(particle.x, particle.y, particle.z, s=100, label=particle.label)
        else:
            ax.scatter(particle.x, particle.y, particle.z, s=100)
    asteroid_line, = ax.plot([], [], [], color='red', label="Asteroid Path")
    ax.set_title('Asteroid Path in Solar System')
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([-50, 50])

    def update(frame):
        ax.clear()
        ax.scatter(0, 0, 0, c='yellow', s=300, label="Sun")
        for particle in sim.particles[1:]:
            if hasattr(particle, 'label'):
                ax.scatter(particle.x, particle.y, particle.z, s=100, label=particle.label)
            else:
                ax.scatter(particle.x, particle.y, particle.z, s=100)
        ax.plot(x_asteroid[:frame], y_asteroid[:frame], z_asteroid[:frame], color='red')
        ax.set_title('Asteroid Path in Solar System')
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])
        ax.set_zlim([-50, 50])
        return asteroid_line,

    ani = animation.FuncAnimation(fig, update, frames=range(0, len(x_asteroid), 10), interval=50, blit=False)
    ani.save(filename, writer='ffmpeg', fps=30)

asteroid_position = [3, 3, 1.22]
asteroid_velocity = [0.0, 1.1, 0.0]
simulation = initialize_simulation()
x_asteroid, y_asteroid, z_asteroid = simulate_asteroid_path(asteroid_position, asteroid_velocity, simulation, duration=1000)
animate_simulation(x_asteroid, y_asteroid, z_asteroid, simulation, filename="asteroid_simulation.mp4")
