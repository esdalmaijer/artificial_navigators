#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy

# Deteminde the parameter range.
parameter_range = "revision"

# Simulation settings.
simulation_types = ["experimental", "solo", "pair"]
n_repetitions = 50
n_generations = 5
n_flights_per_generation = 12

# Simulation parameter ranges.
if parameter_range == "wide":
    p_goal_range = numpy.arange(0.1, 0.75, 0.05)
    p_goal_range = list(numpy.round(p_goal_range, 3))
    p_memory_range = numpy.arange(0.1, 0.75, 0.05)
    p_memory_range = list(numpy.round(p_memory_range, 3))
    p_social_range = numpy.arange(0.1, 0.75, 0.05)
    p_social_range = list(numpy.round(p_social_range, 3))
elif parameter_range == "narrow":
    p_goal_range = numpy.arange(0.025, 0.36, 0.025)
    p_goal_range = list(numpy.round(p_goal_range, 3))
    p_memory_range = numpy.arange(0.025, 0.51, 0.025)
    p_memory_range = list(numpy.round(p_memory_range, 3))
    p_social_range = numpy.arange(0.025, 0.36, 0.025)
    p_social_range = list(numpy.round(p_social_range, 3))
elif parameter_range == "revision":
    p_goal_range = numpy.arange(0.05, 0.36, 0.05)
    p_goal_range = [0.01, 0.025] + list(numpy.round(p_goal_range, 3))
    p_memory_range = numpy.arange(0.05, 0.51, 0.05)
    p_memory_range = list(numpy.round(p_memory_range, 3))
    p_social_range = numpy.arange(0.05, 0.36, 0.05)
    p_social_range = list(numpy.round(p_social_range, 3))
elif parameter_range == "test":
    p_goal_range = [0.2]
    p_memory_range = [0.25]
    p_social_range = [0.4]
elif "control" in parameter_range:
    p_goal_range =   [0.15]
    p_memory_range = [0.4]
    p_social_range = [0.2]
else:
    raise Exception("Parameter range '{}' is not a valid input!".format( \
        parameter_range))

sd_goal_range = [1.0]
sd_continuity = 0.35
sd_social = 0.8
sd_memory_max_range = [0.9]
sd_memory_min = 0.4
sd_memory_steps = 5

if parameter_range == "control_precise-memory":
    sd_memory_max_range = [1e-6]
    sd_memory_min = 1e-6
    sd_memory_steps = 1
elif parameter_range == "control_memory-precision-02":
    sd_memory_min = 0.2
elif parameter_range == "control_memory-precision-08":
    sd_memory_min = 0.8
elif parameter_range == "control_goal-precision":
    sd_goal_range = [0.1, 0.5, 2.0]
elif parameter_range == "control_social-precision-01":
    sd_social = 0.1
elif parameter_range == "control_social-precision-04":
    sd_social = 0.4
elif parameter_range == "control_social-precision-16":
    sd_social = 1.6
elif parameter_range == "control_continuity-precision-005":
    sd_continuity = 0.05
elif parameter_range == "control_continuity-precision-0175":
    sd_continuity = 0.175
elif parameter_range == "control_continuity-precision-07":
    sd_continuity = 0.7
elif parameter_range == "control_memory-precision":
    sd_memory_max_range = [0.4, 1.0, 4.0]
elif parameter_range == "control_noise-goal":
    p_goal_range =   [0.15, 0.3]
    p_memory_range = [0.4, 0.05]
    p_social_range = [0.2, 0.15]
elif parameter_range == "control_noise-memory":
    p_goal_range =   [0.15, 0.3]
    p_memory_range = [0.4, 0.05]
    p_social_range = [0.2, 0.15]
elif parameter_range == "control_noise-social":
    p_goal_range =   [0.15, 0.3]
    p_memory_range = [0.4, 0.05]
    p_social_range = [0.2, 0.15]
elif parameter_range == "control_noise-continuity":
    p_goal_range =   [0.15, 0.3]
    p_memory_range = [0.4, 0.05]
    p_social_range = [0.2, 0.15]
elif parameter_range == "control_post-generation-goal-noise":
    p_goal_range =   [0.15, 0.3]
    p_memory_range = [0.4, 0.05]
    p_social_range = [0.2, 0.15]

# Compute the range of P(memory) parameter values that will be used, and count
# the number of simulations that the current settings will result in.
n_simulations = 0
p_continuity_range_used = []
for p_goal in p_goal_range:
    for p_social in p_social_range:
        for p_memory in p_memory_range:
            p_continuity = round(1.0 - (p_goal + p_social + p_memory), 3)
            p_total = round(p_goal + p_social + p_memory + p_continuity, 3)
            if (0 <= p_continuity <= 1) and (p_total == 1):
                n_simulations += 1
                if p_continuity not in p_continuity_range_used:
                    p_continuity_range_used.append(p_continuity)
p_continuity_range_used.sort()
n_simulations = n_simulations * len(sd_goal_range) * len(sd_memory_max_range)

# Path settings.
start_pos = (0.0, 0.0)
goal_pos = (30.0, 100.0)
n_landmarks = 10

# We use to set the start heading towards the goal, but we now just set it to
# North (heading of 0 radians).
start_heading = 0.0

# Set the stepsize (distance travelled per time unit).
stepsize = 1.0

# Set the mean and standard deviation for the distribution around which the
# transition between alignment and convergence of headings. Alignment occurs
# at short distances, when agents can see each other well. Convergence occurs
# at further distances, when agents can only see each other's location. The
# distance at which this transition happens is determined by the variable 
# alignment_distance_mean, and the variation around this is set with the 
# variable alignment_distance_sd.
alignment_distance_mean = 0.5
alignment_distance_sd = 0.1

# The maximum number of steps is based on the Euclidean distance between start 
# and goal, and allows for a factor of 24 above that.
d = numpy.sqrt((start_pos[0]-goal_pos[0])**2 + (start_pos[1]-goal_pos[1])**2)
max_steps = int(numpy.ceil((d / stepsize) * 24))

# Set the goal threshold relatively close to the goal, but not too narrowly.
goal_threshold = 4.0 * stepsize
# Set the landmark threshold relatively widely. This corresponds to being able
# to see the landmark from afar.
landmark_threshold = 10.0 * stepsize

# Construct landmarks as a sparse(ish) collection of points scattered across
# the map.
xspan = abs(goal_pos[0] - start_pos[0])
yspan = abs(goal_pos[1] - start_pos[1])
buffer = 0.5 * max(xspan, yspan)
xmin = min(goal_pos[0], start_pos[0]) - buffer
xmax = max(goal_pos[0], start_pos[0]) + buffer
x = numpy.arange(xmin, xmax, stepsize)
ymin = min(goal_pos[1], start_pos[1]) - buffer
ymax = max(goal_pos[1], start_pos[1]) + buffer
y = numpy.arange(ymin, ymax, stepsize)

xx, yy = numpy.meshgrid(x, y)
flat_indices = numpy.arange(0, xx.size, 1)
rng = numpy.random.default_rng(seed=19)
sel = rng.choice(flat_indices, size=int(numpy.ceil(xx.size*0.25)), \
    replace=False)

landmarks = {"x":xx.flatten()[sel], "y":yy.flatten()[sel]}
