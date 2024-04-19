#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import math

import numpy
import scipy.special
from scipy.stats import vonmises

# Some circular stats helper function.
def abs_circular_distance(a, b):
    """Computed the absolute circular distance between angles (in radians).
    """

    # Wrap the angles.
    a = a % (2*numpy.pi)
    b = b % (2*numpy.pi)

    # Compute the absolute angular distance between the angles.
    d = numpy.pi - numpy.abs(numpy.pi - numpy.abs(a-b))

    return d

def circular_distance(a, b):
    """Computed the signed circular distance between angles (in radians).
    Negative values indicate counter-clockwise distances.
    """

    # Wrap the angles.
    a = a % (2*numpy.pi)
    b = b % (2*numpy.pi)

    # Compute the angular distance between the angles, with positive valus
    # indicating clockwise and negative values indicating counter-clockwise
    # distances.
    d = b - a
    # Convert to NumPy array if necessary.
    reconvert = False
    if type(d) not in [numpy.array, numpy.ndarray]:
        d = numpy.array([d])
        reconvert = True
    # Compute the signed distance.
    d[d>numpy.pi] = -1 * (2*numpy.pi - d[d>numpy.pi])
    d[d<=-numpy.pi] = (2*numpy.pi + d[d<=-numpy.pi])
    # Reconvert to a single value if necessary.
    if reconvert:
        d = d[0]

    return d

def circular_mean(a, weights=None):
    """Computes the average of a set of angles (in radians) by casting them
    into cartesian space, computing means independently, and then converting
    back into an angle.
    """
    if weights is None:
        m_sin = numpy.mean(numpy.sin(a))
        m_cos = numpy.mean(numpy.cos(a))
    else:
        m_sin = numpy.mean(numpy.sin(a) * weights)
        m_cos = numpy.mean(numpy.cos(a) * weights)
    
    return numpy.arctan2(m_sin, m_cos)
    
def kappa2sd(kappa):
    """Converts a kappa value of a Von Mises distribution to the equivalent
    standard deviation of the corresponding normal distribution.
    """

    if type(kappa) not in [numpy.array, numpy.ndarray]:
        kappa = numpy.array([kappa])
    
    zero = kappa == 0
    infinite = kappa == numpy.inf
    other =  numpy.invert(zero | infinite)
    
    sd = numpy.zeros(kappa.shape)
    sd[zero] = numpy.inf
    sd[infinite] = 0.0
    sd[other] = numpy.sqrt(-2.0 * numpy.log( \
        scipy.special.iv(1.0, kappa[other]) \
        / scipy.special.iv(0.0, kappa[other])))

    if (len(sd.shape)==1) & (sd.shape[0]==1):
        return sd[0]
    else:
        return sd

def sd2kappa(sd):
    """Converts a standard deviation of a normal distribution to the
    equivalent kappa value of the corresponding Von Mises distribution.
    """
    
    if type(sd) not in [numpy.array, numpy.ndarray]:
        sd = numpy.array([sd])
    
    r = numpy.exp((-1*(sd)**2) / 2)
    k = 1 / (r**3 - 4*r**2 + 3*r)
    selone = numpy.array(r < 0.85)
    k[selone] = -0.4 + 1.39 * r[selone] + 0.43/(1 - r[selone])
    seltwo = numpy.array(r < 0.53)
    k[seltwo] = 2 * r[seltwo] + r[seltwo]**3 + (5 * r[seltwo]**5)/6;
    
    if (len(k.shape)==1) & (k.shape[0] == 1):
        return k[0]
    else:
        return k

# Self-moving agent.
class Agent:
    
    def __init__(self, starting_heading=0.0, starting_position=(0.0, 0.0), \
        goal_position=(80.0, 60.0), p_goal=0.05, sd_goal=1.0, \
        p_social=0.03, sd_social=1.0, p_continuity=0.02, sd_continuity=0.5, \
        p_memory=0.9, sd_memory=0.5, stepsize=1.0, goal_threshold=4.0, \
        landmark_threshold=10.0, n_landmarks=20, landmarks=None, \
        prefer_first_path=True, alignment_distance_mean=0.5, \
        alignment_distance_sd=0.1, goal_lesion=False, social_lesion=False, \
        memory_lesion=False, continuity_lesion=False):
        
        self._start_heading = copy.deepcopy(starting_heading)
        self._heading_prev = copy.deepcopy(starting_heading)
        self._heading = copy.deepcopy(starting_heading)
        self._start_position = copy.deepcopy(starting_position)
        self._position = copy.deepcopy(starting_position)
        self._goal = copy.deepcopy(goal_position)
        
        self._goal_p = p_goal
        self._goal_sd = sd_goal
        self._goal_kappa = sd2kappa(self._goal_sd)
        self._goal_lesion = goal_lesion
        
        self._social_p = p_social
        self._social_sd = sd_social
        self._social_kappa = sd2kappa(self._social_sd)
        self._social_lesion = social_lesion

        self._continuity_p = p_continuity
        self._continuity_sd = sd_continuity
        self._continuity_kappa = sd2kappa(self._continuity_sd)
        self._continuity_lesion = continuity_lesion
        
        self._memory_p = p_memory
        self._memory_sd = sd_memory
        if type(sd_memory) in [tuple, list, numpy.ndarray]:
            if len(sd_memory) == 3:
                sd_memory = numpy.linspace(sd_memory[0], sd_memory[1], \
                    sd_memory[2])
            self._memory_kappa = [sd2kappa(sd) for sd in sd_memory]
        else:
            self._memory_kappa = sd2kappa(self._memory_sd)
        self._memory_lesion = memory_lesion

        self._current_landmark = None
        self._n_landmarks = n_landmarks
        self._landmarks = landmarks

        self._stepsize = stepsize
        self._goal_threshold = goal_threshold
        self._landmark_threshold = landmark_threshold
        self._heading_weights = numpy.array([ \
            self._goal_p, \
            self._social_p, \
            self._memory_p, \
            self._continuity_p, \
            ])
        
        self._alignment_distance_mean = alignment_distance_mean
        self._alignment_distance_sd = alignment_distance_sd
        
        self._finished = True
        self._n_finished = 0
        self._prefer_first_path = prefer_first_path
        self._current_path = {"x":[], "y":[]}
        self._path_history = []
        self._preferred_path = None
        self._preferred_path_finished = False
    
    def advance_position(self, other_position=None, other_heading=None, \
        force_position=None):
        
        # Only move if we're not already at the goal.
        if not self._finished:

            # Compute distance to the goal.
            goal_dist = numpy.sqrt((self._position[0]-self._goal[0])**2 \
                + (self._position[1]-self._goal[1])**2)

            # Finish if we're at the goal now.
            if goal_dist < self._goal_threshold:
                # Set the current position to the goal.
                new_x, new_y = self._goal
                self._finished = True
                self._n_finished += 1

            # Update the heading and position if we're not there yet.
            else:
                # Force the requested position.
                if force_position is not None:
                    # Grab the new position values.
                    new_x = force_position[0]
                    new_y = force_position[1]
                    # Manually compute and update the heading.
                    new_heading = math.atan2(new_y-self._position[1], \
                        new_x-self._position[0])
                    self.set_heading(new_heading)
                else:
                    # Compute the new heading.
                    self.update_heading(other_position=other_position, \
                        other_heading=other_heading)
                    # Advance in the direction of the heading.
                    new_x = self._position[0] + self._stepsize \
                        * math.cos(self._heading)
                    new_y = self._position[1] + self._stepsize \
                        * math.sin(self._heading)

            # Update the current position.
            self.set_position((new_x,new_y))

            # Store the new position in the path history.
            self._current_path["x"].append(new_x)
            self._current_path["y"].append(new_y)
            
            # Check if the new position is within distance of a landmark, but
            # only do so if we don't have a memorised path yet.
            if (not self._preferred_path_finished) \
                and (type(self._landmarks)==dict):
                # Compute the distance between all landmarks and the current
                # position.
                d_landmark = numpy.sqrt((self._landmarks["x"] - new_x)**2 \
                    + (self._landmarks["y"] - new_y)**2)
                # Find the shortest distance.
                i = numpy.argmin(d_landmark)
                # If the distance is shorter than the landmark threshold, add
                # the landmark to the path memory.
                if d_landmark[i] < self._landmark_threshold:
                    # If this is the first landmark, create new lists for it.
                    if self._preferred_path is None:
                        self._preferred_path = { \
                            "x":[self._landmarks["x"][i]], \
                            "y":[self._landmarks["y"][i]], \
                            }
                    # Only add the landmark if it isn't equal to the previous
                    # landmark (otherwise we would simply keep adding the same
                    # landmark while flying in its vicinity).
                    else:
                        if (self._preferred_path["x"][-1] \
                            != self._landmarks["x"][i]) \
                            or (self._preferred_path["y"][-1] \
                            != self._landmarks["y"][i]):
                            # Add the current landmark to the preferred path.
                            self._preferred_path["x"].append( \
                                self._landmarks["x"][i])
                            self._preferred_path["y"].append( \
                                self._landmarks["y"][i])

            # If we just reached the finish, save the path into the history.
            if self._finished:
                # If we have a preferred path and this is the first journey,
                # the path history should start with the preferred path.
                if (self._preferred_path is not None) and \
                    (not self._preferred_path_finished):
                    self._preferred_path["x"] = \
                        numpy.array(self._preferred_path["x"])
                    self._preferred_path["y"] = \
                        numpy.array(self._preferred_path["y"])
                    self._preferred_path_finished = True
                # Reduce the path to a select number of landmarks.
                else:
                    sel = numpy.linspace(0, len(self._current_path["x"])-1, \
                        self._n_landmarks, dtype=numpy.int64)
                    shape = (2, self._n_landmarks)
                    path_array = numpy.zeros(shape, dtype=numpy.float64)
                    path_array[0,:] = numpy.array(self._current_path["x"])[sel]
                    path_array[1,:] = numpy.array(self._current_path["y"])[sel]
                    self._path_history.append(path_array)
        
        return self.get_position(), self._finished
    
    def compute_heading_without_error(self, other_position=None):

        # Assign weights to each heading component.
        w = numpy.copy(self._heading_weights)

        # Compute goal heading component.
        hg = self.sample_goal_heading(no_error=True)
        
        # Eliminate the effect of the social heading component if there is no 
        # other agent.
        if (other_position is None) or (other_position[0]==(None,None)):
            hs = 0.0
            w[1] = 0.0
            w /= numpy.sum(w)
        # Compute the heading towards the other agent (in actuality, this is
        # the heading towards the position where the agent was just 
        # previously).
        else:
            # Compute the heading towards another agent.
            hs = self.sample_heading(other_position, self._social_kappa, \
                no_error=True)
        
        # Compute path memory component.
        if len(self._n_finished) > 0:
            hm = self.sample_memory_heading(no_error=True)
        # Eliminate the memory heading if no memory exists yet.
        else:
            hm = 0.0
            w[2] = 0.0
            w /= numpy.sum(w)
        
        # Compute heading continuity component.
        hc = self.sample_continuity(no_error=True)
        
        # Compute the new heading as a weighted average.
        return circular_mean([hg, hs, hm, hc], weights=w)
    
    def finished(self):
        return self._finished
    
    def get_current_distributions(self, other_position=None, \
        other_heading=None, heading_centre=False, next_heading=None):
        
        # Create x values for a circular normal distribution.
        if not heading_centre:
            x = numpy.linspace(0, 2*numpy.pi, 360)
        
        # If a prospective heading was not provided, assume we'll stay on the.
        if next_heading is not None:
            next_heading = self._heading
            prev_heading = self._heading_prev
        else:
            prev_heading = self._heading

        # Goal distribution.
        # Compute the heading towards the goal from the current position, and
        # then create a Von Mises distribution centred on the goal.
        gh = self.sample_heading(self._goal, self._goal_kappa, no_error=True)
        if heading_centre:
            gd = circular_distance(gh, self._heading)
            goal = scipy.stats.vonmises.pdf(gd, loc=0.0, \
                kappa=self._goal_kappa)
        else:
            goal = scipy.stats.vonmises.pdf(x, loc=gh, kappa=self._goal_kappa)
        
        # Social distribution.
        # If neither location or position are given, the social distribution
        # is uniform.
        if (other_position is None) and (other_heading is None):
            if heading_centre:
                sd = numpy.nan
                social = 1.0 / (2*numpy.pi)
            else:
                social = numpy.ones(x.shape[0]) / (2*numpy.pi)
        # If other position (and heading) are given, we can compute the
        # social heading.
        else:
            # Compute the heading towards the other agent, and create a Von Mises
            # distribution around it.
            oh = self.sample_heading(other_position, self._social_kappa, \
                no_error=True)
            if heading_centre:
                cd = circular_distance(oh, next_heading)
                convergence = scipy.stats.vonmises.pdf(cd, loc=0.0, \
                    kappa=self._social_kappa)
            else:
                convergence = scipy.stats.vonmises.pdf(x, loc=oh, \
                    kappa=self._social_kappa)
            # If another agent's heading is not given, the heading towards their
            # position is all we have.
            if other_heading is None:
                social = convergence
            # If another agent's heading is given, then try to align with that
            # heading by sampling from a distribution centred on it. Noise is
            # mixed in as a function of distance from the other agent's 
            # location.
            else:
                # Create a Von Mises distribution around the other heading.
                if heading_centre:
                    ad = circular_distance(other_heading, next_heading)
                    alignment = scipy.stats.vonmises.pdf(ad, loc=0.0, \
                        kappa=self._social_kappa)
                else:
                    alignment = scipy.stats.vonmises.pdf(x, \
                        loc=other_heading, kappa=self._social_kappa)
                # Compute the distance to the other agent's position.
                d = numpy.sqrt((self._position[0]-other_position[0])**2 \
                    + (self._position[1]-other_position[1])**2)
                # Compute the proportion of the result that should be based
                # on the other agent's location. This increases after a certain
                # distance, before which we mostly try to align headings with
                # agents who are close by.
                p = scipy.stats.norm.cdf(d, \
                    loc=self._alignment_distance_mean, \
                    scale=self._alignment_distance_sd)
                # Mix the true heading and heading towards the other agent.
                social = (1-p)*alignment + p*convergence
                # Combine the circular distances.
                if heading_centre:
                    sd = circular_mean(numpy.array([cd, ad]), \
                        weights=numpy.array([p, 1-p]))
        
        # Memory distribution.
        # If we don't have a memorised path yet, the memory distribution is
        # a uniform distribution.
        if not self._preferred_path_finished:
            if heading_centre:
                md = numpy.nan
                memory = 1.0 / (2*numpy.pi)
            else:
                memory = numpy.ones(x.shape[0]) / (2*numpy.pi)
        else:
            # Compute the distance to the current target landmark.
            if self._current_landmark >= self._preferred_path["x"].shape[0]:
                ref_point = self._goal
            else:
                ref_point = \
                    (self._preferred_path["x"][self._current_landmark], \
                    self._preferred_path["y"][self._current_landmark])
            dx = self._position[0] - ref_point[0]
            dy = self._position[1] - ref_point[1]
            d = numpy.sqrt(dx**2 + dy**2)
            # If the distance is smaller than the threshold, update the
            # current landmark to the next one along the path. Also make 
            # sure we do not exceed the number of landmarks.
            if d < self._landmark_threshold:
                self._current_landmark += 1
            # Get the next landmark's coordinates, or the goal's 
            # coordinates if we passed the last landmark.
            if self._current_landmark >= self._preferred_path["x"].shape[0]:
                next_point = self._goal
            else:
                next_point = \
                    (self._preferred_path["x"][self._current_landmark], \
                    self._preferred_path["y"][self._current_landmark])
            # If memory precision is not constant, choose the most
            # appropriate kappa.
            if type(self._memory_kappa) == list:
                if self._n_finished >= len(self._memory_kappa):
                    kappa = self._memory_kappa[-1]
                else:
                    kappa = self._memory_kappa[self._n_finished-1]
            else:
                kappa = self._memory_kappa
            # Compute the heading towards the current target landmark.
            mh = self.sample_heading(next_point, kappa, no_error=True)
            if heading_centre:
                md = circular_distance(mh, next_heading)
                memory = scipy.stats.vonmises.pdf(md, loc=0.0, kappa=kappa)
            else:
                memory = scipy.stats.vonmises.pdf(x, loc=mh, kappa=kappa)
        
        # Heading distribution.
        # Create a Von Mises distribution centred on the current heading.
        if heading_centre:
            cd = next_heading - prev_heading
            continuity = scipy.stats.vonmises.pdf(cd, loc=0.0, \
                kappa=self._continuity_kappa)
        else:
            continuity = scipy.stats.vonmises.pdf(x, loc=self._heading, \
                kappa=self._continuity_kappa)
        
        # Create a dictionairy with the combined components.
        components = { \
            "goal": self._goal_p * goal, \
            "social": self._social_p * social, \
            "memory": + self._memory_p * memory, \
            "continuity": self._continuity_p * continuity, \
            }

        # Compute the combined probability density function.
        pdf = components["goal"] + components["social"] \
            + components["memory"] + components["continuity"]
        
        if heading_centre:
            d = { \
                "goal": gd, \
                "social": sd, \
                "memory": + md, \
                "continuity": cd, \
                }
            return pdf, components, d
        
        return pdf, components
    
    def get_heading(self):
        return self._heading

    def get_goal_position(self):
        return self._goal

    def get_position(self):
        return self._position
    
    def get_speed(self):
        if self._finished:
            return 0.0
        else:
            return self._stepsize
    
    def expected_next_position(self):
        
        # Get the current position, heading, and speed.
        pos = self.get_position()
        heading = self.get_heading()
        speed = self.get_speed()
        # Compute the expected next position on the basis of where the agent
        # is likely to end up based on the current heading and speed.
        return (pos[0] + speed * numpy.cos(heading), \
            pos[1] + speed * numpy.sin(heading))
    
    def release(self, starting_position=None):
        
        # Only allow release if we're not currently flying.
        if not self._finished:
            raise Exception("Can't release agent; they are still en route.")

        # Reset everything to starting point.
        if starting_position is None:
            self._position = copy.deepcopy(self._start_position)
        else:
            self._position = copy.deepcopy(starting_position)
        self._finished = False
        self._current_path = {"x":[], "y":[]}
        self._current_landmark = 0
    
    def reset(self):
        self._position = copy.deepcopy(self._start_position)
        self._heading_prev = copy.deepcopy(self._start_heading)
        self._heading = copy.deepcopy(self._start_heading)
        self._finished = True
        self._current_path = {"x":[], "y":[]}
        self._current_landmark = 0
        if not self._preferred_path_finished:
            self._preferred_path = None
        
    def sample_continuity(self, compute_probability=False, no_error=False):
        if no_error:
            return self._heading
        else:
            return numpy.random.vonmises(self._heading, \
                self._continuity_kappa, size=None)
    
    def sample_goal_heading(self, no_error=False):
        return self.sample_heading(self._goal, self._goal_kappa, \
            no_error=no_error)
    
    def sample_heading(self, target_position, target_kappa, no_error=False):
        # Start by setting the current position to (0,0)
        dx = target_position[0] - self._position[0]
        dy = target_position[1] - self._position[1]
        # Compute the direction towards the goal from the current position.
        d = math.atan2(dy, dx)
        # Sample the new goal-direction, using the current direction towards 
        # the goal and random error (or without error, if requested).
        if no_error:
            target_heading = d
        else:
            target_heading = numpy.random.vonmises(d, target_kappa, size=None)

        return target_heading
    
    def sample_memory_heading(self, no_error=False):
        
        # Only rely on memory if there is a path in memory.
        if not self._preferred_path_finished:
            return self.sample_random_heading()
        # Find the heading towards the nearest next landmark.
        else:
            # Compute the distance to the current target landmark.
            if self._current_landmark >= self._preferred_path["x"].shape[0]:
                ref_point = self._goal
            else:
                ref_point = \
                    (self._preferred_path["x"][self._current_landmark], \
                    self._preferred_path["y"][self._current_landmark])
            dx = self._position[0] - ref_point[0]
            dy = self._position[1] - ref_point[1]
            d = numpy.sqrt(dx**2 + dy**2)
            # If the distance is smaller than the threshold, update the
            # current landmark to the next one along the path. Also make 
            # sure we do not exceed the number of landmarks.
            if d < self._landmark_threshold:
                self._current_landmark += 1
            # Get the next landmark's coordinates, or the goal's 
            # coordinates if we passed the last landmark.
            if self._current_landmark >= self._preferred_path["x"].shape[0]:
                next_point = self._goal
            else:
                next_point = \
                    (self._preferred_path["x"][self._current_landmark], \
                    self._preferred_path["y"][self._current_landmark])
            # If memory precision is not constant, choose the most
            # appropriate SD.
            if type(self._memory_kappa) == list:
                if self._n_finished >= len(self._memory_kappa):
                    kappa = self._memory_kappa[-1]
                else:
                    kappa = self._memory_kappa[self._n_finished-1]
            else:
                kappa = self._memory_kappa
            # Compute the heading towards the current target landmark.
            return self.sample_heading(next_point, kappa, \
                no_error=no_error)
    
    def sample_random_heading(self):
        
        return numpy.random.rand() * 2 * numpy.pi
    
    def sample_social_heading(self, other_position=None, other_heading=None):

        # If neither location or position are given, we can't compute a 
        # social heading.
        if (other_position is None) and (other_heading is None):
            social_heading = None
        # If another agent's heading is not given, then adjust towards its
        # location by computing the heading where they are.
        if other_heading is None:
            social_heading = self.sample_heading(other_position, \
                self._social_kappa)
        # If another agent's heading is given, then try to align with that
        # heading by sampling from a distribution centred on it. Noise is
        # mixed in as a function of distance from the other agent's 
        # location.
        else:
            # Sample from a Von Mises centred on the other heading.
            alignment = numpy.random.vonmises(other_heading, \
                self._social_kappa, size=None)
            # Sample from a Von Mises distribution towards the other agent.
            convergence = self.sample_heading(other_position, \
                self._social_kappa)
            # Compute the distance to the other agent's position.
            d = numpy.sqrt((self._position[0]-other_position[0])**2 \
                + (self._position[1]-other_position[1])**2)
            # Compute the proportion of the result that should be based
            # on the other agent's location. This increases after a certain
            # distance, before which we mostly try to align headings with
            # agents who are close by.
            p = scipy.stats.norm.cdf(d, loc=self._alignment_distance_mean, \
                scale=self._alignment_distance_sd)
            # Mix the true heading and heading towards the other agent.
            social_heading = circular_mean([alignment, convergence], \
                weights=[1-p, p])
        
        return social_heading

    def set_heading(self, new_heading):
        self._heading_prev = copy.copy(self._heading)
        self._heading = new_heading

    def set_goal_position(self, new_goal_position):
        self._goal = new_goal_position

    def set_position(self, new_position):
        self._position = new_position
    
    def update_heading(self, other_position=None, other_heading=None):

        # Assign weights to each heading component.
        w = numpy.copy(self._heading_weights)

        # Compute goal heading component.
        if self._goal_lesion:
            hg = self.sample_random_heading()
        else:
            hg = self.sample_goal_heading()
        
        # Eliminate the effect of the social heading component if there is no 
        # other agent. (Replace the Von Mises distribution by a uniform one.)
        if other_position is None:
            hs = self.sample_random_heading()
        # Compute the heading towards the other agent (and/or the other
        # agent's heading).
        else:
            if self._social_lesion:
                hs = self.sample_random_heading()
            else:
                hs = self.sample_social_heading( \
                    other_position=other_position, \
                    other_heading=other_heading)
        
        # Compute path memory component.
        if self._memory_lesion:
            hm = self.sample_random_heading()
        else:
            hm = self.sample_memory_heading()
        
        # Compute heading continuity component.
        if self._continuity_lesion:
            hc = self.sample_random_heading()
        else:
            hc = self.sample_continuity()
        
        # Compute the new heading as a weighted average.
        new_heading = circular_mean([hg, hs, hm, hc], weights=w)
        self.set_heading(new_heading)
