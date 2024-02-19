import matplotlib.pyplot as plt
import seaborn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy import array

from niopy.geometric_transforms import dst, nrm
from scopy.linear_fresnel import uniform_centers, Heliostat, PrimaryField, Absorber, primaries_curvature_radius
from scopy.nio_concentrators import symmetric_cec2evacuated_tube, cpc_type
from utils import plot_line


# Configuring plots #######################################################################
seaborn.set(style='whitegrid')
seaborn.set_context('notebook')

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = '\\usepackage{amsmath}\n \\usepackage{amssymb}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'NewComputerModern10'

# Main geometrical data ############################################

# primary field data
mirror_width = 750
total_primary_width = 16560
nbr_mirrors = 14

# receiver data
receiver_height = 7200
tube_radius = 35
outer_cover_radius = 62.5
inner_cover_radius = outer_cover_radius - 3.
####################################################################

# The receiver #########################################################################################

#  absorber
absorber = Absorber.evacuated_tube(name='evacuated_tube',
                                   center=array([0, receiver_height]),
                                   absorber_radius=tube_radius,
                                   outer_cover_radius=outer_cover_radius,
                                   inner_cover_radius=inner_cover_radius)

# the secondary optic
secondary_sections = symmetric_cec2evacuated_tube(tube_center=absorber.absorber_tube.center,
                                                  tube_radius=absorber.absorber_tube.radius,
                                                  cover_radius=absorber.outer_radius,
                                                  source_distance=receiver_height,
                                                  source_width=total_primary_width,
                                                  nbr_pts=50, upwards=False, dy=0)

secondary_optic = cpc_type(left_conic=secondary_sections[0],
                           left_involute=secondary_sections[1],
                           right_involute=secondary_sections[2],
                           right_conic=secondary_sections[3])

# aim-point at secondary optic aperture
rec_aim = secondary_optic.aperture_center
########################################################################################################

# The primary field ####################################################################################

# center points of the primary mirror
primaries_center = uniform_centers(mirror_width=mirror_width,
                                   total_width=total_primary_width,
                                   nbr_mirrors=nbr_mirrors)

# curvature radius of the primary mirrors
radii_values = primaries_curvature_radius(centers=primaries_center,
                                          rec_aim=rec_aim,
                                          curvature_design='zenithal')

# primary mirrors as 'Heliostat' objects
heliostats = [Heliostat(center=hc, width=mirror_width, radius=r)
              for hc, r in zip(primaries_center, radii_values)]

# the primary field as a 'PrimaryField' object
primary_field = PrimaryField(heliostats=heliostats)
########################################################################################################

fig = plt.figure(figsize=(12, 5), dpi=300)

ax = fig.add_subplot(1, 2, 1)
plt.title('(a) General view')
plt.xlabel('$x$ [mm]')
plt.ylabel('$z$ [mm]')

ax.plot(*absorber.absorber_tube.contour.T, color='red')
ax.plot(*absorber.outer_tube.contour.T, color='orange', ls='dashed')
ax.plot(*absorber.inner_tube.contour.T, color='orange', ls='dashed')

plt.plot(*secondary_optic.contour.T, color='blue')

[plt.plot(*plot_line(hc, rec_aim), color='green', lw=0.75) for hc in primaries_center]

[plt.plot(*plot_line(hc, hc + 0.8*absorber.center), color='orange', lw=0.75) for hc in primaries_center]
primary_field.plot_primaries(theta_t=0., rec_aim=rec_aim, support_size=500)
plt.legend()

ax = fig.add_subplot(1, 2, 2)
plt.title('(b) Receiver details')
plt.xlabel('$x$ [mm]')

ax.plot(*absorber.absorber_tube.contour.T, color='red', label='Absorber tube')
ax.plot(*absorber.outer_tube.contour.T, color='orange', ls='dashed', label='Glass cover')
ax.plot(*absorber.inner_tube.contour.T, color='orange', ls='dashed')
plt.plot(*secondary_optic.contour.T, color='blue', label='Secondary optic')
plt.legend()

plt.xlim([absorber.center[0] - 250, absorber.center[0] + 250])
plt.ylim([absorber.center[1] - 250, absorber.center[1] + 250])

plt.tight_layout()
plt.show()

s1, s2 = secondary_optic.contour[0], secondary_optic.contour[-1]
flat_absorber = Absorber.flat(width=dst(s1, s2), center=rec_aim, axis=nrm(s2-s1))

transversal_if, longitudinal_if = primary_field.optical_analysis(flat_absorber=flat_absorber,
                                                                 rec_aim=rec_aim, symmetric=True)

gama0 = transversal_if[0][1]

fig = plt.figure(dpi=300)
ax = fig.add_subplot()

plt.title('Factorized Incidence Angle Modifiers')
ax.plot(transversal_if.T[0], transversal_if.T[-1] / gama0, label='Transversal')
ax.plot(longitudinal_if.T[0], longitudinal_if.T[-1] / gama0, label='Longitudinal')

plt.xlabel('Incidence angle [deg]')
plt.ylabel('IAM')

plt.legend()

plt.tight_layout()
plt.show()
