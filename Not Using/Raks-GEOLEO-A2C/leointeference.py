import turtle
import pathloss, geo, leo
import math
import numpy as np



def leo_user_interference(user, leo_beam, sat_altitude, int_sat_altitude, int_satellite_properties, geo_beams,
                          leo_freq_sub, geo_freq_sub):
    # system noise = kTB = Boltzman constant * Noise temperature of source * Noise bandwidth (Hz)
    thermal_noise = 293 * 0.2e6 * 1.39e-23

    # list of leo users affected with interference
    leo_user_with_interference = []

    geo_interference = 0
    leo_to_leo_interference = 0

    if user.distance(leo_beam) < 60:
        leo_to_user_distance = user.distance(leo_beam)
        leo_rx_to_leo_user = pathloss.leo_to_user_rx(altitude=sat_altitude, distance=leo_to_user_distance)

    else:
        leo_rx_to_leo_user =0


    # LEO to LEO interference calculation
    for leo_interference_beam in range(len(int_satellite_properties)):
        if ((leo_beam.fillcolor() == int_satellite_properties[leo_interference_beam].fillcolor()) and \
                (user.distance(int_satellite_properties[leo_interference_beam]) < 60)):
            int_leo_to_user_distance = user.distance(int_satellite_properties[0])
            int_leo_rx_to_leo_user = pathloss.leo_to_user_rx(altitude=int_sat_altitude,
                                                             distance=int_leo_to_user_distance)
            leo_to_leo_interference += int_leo_rx_to_leo_user

            if user not in leo_user_with_interference:
                leo_user_with_interference.append(user)

        else:
            leo_to_leo_interference += 0

    for geo_interference_beam in range(len(geo_beams)):
        if ((leo_freq_sub[leo_beam.fillcolor()][0] in geo_freq_sub[geo_beams[geo_interference_beam].fillcolor()]) and \
                (user.distance(geo_beams[geo_interference_beam]) < 95) and \
                (user.distance(leo_beam) < 60)):
            int_geo_beam_to_user_distance = user.distance(geo_beams[geo_interference_beam])
            int_geo_rx_to_leo_user = pathloss.leo_to_user_rx(altitude=pathloss.geo_satellite_altitude,
                                                             distance=int_geo_beam_to_user_distance)

            geo_interference += int_geo_rx_to_leo_user

    leo_sinr = leo_rx_to_leo_user/(geo_interference + leo_to_leo_interference + thermal_noise)

    return leo_sinr,(geo_interference + leo_to_leo_interference)
