import pathloss, geo, leo, math




def geo_user_interference(user, beam,l1_turtles, l2_turtles, leo_freq_sub, geo_freq_sub, bandwidth_ratio):
    leo_interference = 0

    # system noise = kTB = Boltzman constant * Noise temperature of source * Noise bandwidth (Hz)
    thermal_noise = 293 * 0.2e6 * 1.39e-23

    # list with geo users affected with interference
    geo_user_with_interference = []

    # calculating GEO RX power to GEO users
    geo_beam_user_distance = user.distance(0,0) # calulating the horizontal distance from (0,0) to GEO user
    # geo_beam_user_distance = user.distance(beam)
    geo_rx_to_geo_user = pathloss.geo_to_user_rx(altitude=pathloss.geo_satellite_altitude,
                                                 distance=geo_beam_user_distance)

    # for geo_interference_beam in range(len(geo_beams)):
    #     if (leo_freq_sub[leo_beam.fillcolor()][0] in geo_freq_sub[geo_beams[geo_interference_beam].fillcolor()]) and \
    #             (user.distance(geo_beams[geo_interference_beam]) < 95):

    for i in range(7):  # range(7) due to number of LEO beams in each LEO is 7

        # Iterating through LEO A
        if (leo_freq_sub[l1_turtles[i].fillcolor()][0] in geo_freq_sub[beam.fillcolor()]) and (user.distance(l1_turtles[i]) < 60):
            # calculating LEO RX power to GEO users
            #  *5 -> to match grid to actual distance
            #  -200 -> to calculate actual distance from LEO beam edge to user
            la_h_distance = user.distance(l1_turtles[0])
            la_rx_to_geo_user = pathloss.leo_to_geo_user_rx(altitude=pathloss.leo_satellite_altitude_A,
                                                        distance=la_h_distance, bandwidth_ratio= bandwidth_ratio)
            leo_interference += la_rx_to_geo_user

            if user not in geo_user_with_interference:
                geo_user_with_interference.append(user)

        # Iterating through LEO B
        elif (leo_freq_sub[l2_turtles[i].fillcolor()][0] in geo_freq_sub[beam.fillcolor()]) and (user.distance(l2_turtles[i]) < 60):
            lb_h_distance = user.distance(l2_turtles[0])
            lb_rx_to_geo_user = pathloss.leo_to_geo_user_rx(altitude=pathloss.leo_satellite_altitude_B,
                                                        distance=lb_h_distance, bandwidth_ratio= bandwidth_ratio)
            leo_interference += lb_rx_to_geo_user

            if user not in geo_user_with_interference:
                geo_user_with_interference.append(user)

        else:
            leo_interference += 0

    geo_sinr = geo_rx_to_geo_user/(leo_interference + thermal_noise*bandwidth_ratio)

    return leo_interference, geo_sinr
