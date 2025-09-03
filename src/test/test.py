import VSM.VSM.VSM_forward as VSM_forward

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate a grid
grid_size = 100  # Grid size (100x100)
x = np.linspace(-10, 10, grid_size)
y = np.linspace(-10, 10, grid_size)
x, y = np.meshgrid(x, y)

# Step 2: Define Mogi source parameters
xcen, ycen = 0, 0  # Mogi source at the center
depth = 3  # Random depth (km)
dVol = 1e6  # Random volume change (m^3)
nu = 0.25  # Poisson's ratio

# Step 3: Simulate deformation
ux, uy, uz = VSM_forward.mogi(x, y, xcen, ycen, depth, dVol, nu)

# Step 4: Define geometry
azimuth = 12  # Azimuth angle (degrees)
incident = 45  # Incident angle (degrees)

# Convert angles to radians
azimuth_rad = np.deg2rad(azimuth)
incident_rad = np.deg2rad(incident)

# Step 5: Convert deformation to LOS
# LOS deformation for azimuth = +12°
los_plus = (
    -np.sin(incident_rad) * np.cos(azimuth_rad) * ux
    - np.sin(incident_rad) * np.sin(azimuth_rad) * uy
    + np.cos(incident_rad) * uz
)

# LOS deformation for azimuth = -12°
los_minus = (
    -np.sin(incident_rad) * np.cos(np.pi-azimuth_rad) * ux
    - np.sin(incident_rad) * np.sin(np.pi-azimuth_rad) * uy
    + np.cos(incident_rad) * uz
)

# Step 6: Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Non-converted deformation (uz)
im1 = axes[0].imshow(uz, extent=[-10, 10, -10, 10], cmap='coolwarm')
axes[0].set_title('Non-converted Vertical Deformation (uz)')
axes[0].set_xlabel('x (km)')
axes[0].set_ylabel('y (km)')
plt.colorbar(im1, ax=axes[0])

# LOS deformation for azimuth = +12°
im2 = axes[1].imshow(los_plus, extent=[-10, 10, -10, 10], cmap='coolwarm')
axes[1].set_title('LOS Deformation (+12° Azimuth)')
axes[1].set_xlabel('x (km)')
axes[1].set_ylabel('y (km)')
axes[1].scatter(0, 0, color='black', marker='x', s=100, label='Mogi Source')
plt.colorbar(im2, ax=axes[1])

# LOS deformation for azimuth = -12°
im3 = axes[2].imshow(los_minus, extent=[-10, 10, -10, 10], cmap='coolwarm')
axes[2].set_title('LOS Deformation (-12° Azimuth)')
axes[2].set_xlabel('x (km)')
axes[2].set_ylabel('y (km)')
axes[2].scatter(0, 0, color='black', marker='x', s=100, label='Mogi Source')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()