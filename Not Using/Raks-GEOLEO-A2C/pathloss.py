import math
import parameters
import numpy as np

# GEO system parameters
geo_transmit_power = 300  # in watts
geo_antenna_diameter = 3  # in meters
geo_antenna_efficiency = 0.65

# LEO system parameters
leo_transmit_power = 20  # in watts
leo_antenna_diameter = 0.5  # in meters
leo_antenna_efficiency = 0.65  # in meters

frequency = 1.5e9

user_antenna_diameter = 0.3  # in meters
user_antenna_efficiency = 0.65

geo_satellite_altitude = 35786e3  # in meters
leo_satellite_altitude_A = 600e3  # in meters
leo_satellite_altitude_B = 700e3  # in meters



# Calculate the distance between a satellite and an ground user
def calculate_distance(sat_altitude, horizontal_distance):

    # Calculate the actual distance using Pythagoras theorem "5e3" is from the pixel to m conversion
    distance = math.sqrt((horizontal_distance * 5e3)**2 + sat_altitude**2)

    angle = math.degrees(math.atan(sat_altitude/distance))

    return distance, angle


# Calculate the path loss

def calculate_path_loss(distance, frequency_band, angle):

    # Free space path loss
    path_loss = 20 * math.log10(distance) + 20 * math.log10(frequency_band) - 147.55

    # Rain attenuation (ITU-R P.838-3 model)
    # Mean: 0.0740
    rain_mean = 0.0740
    # Standard Deviation: 0.0080
    rain_std = 0.0080
    rain_intensity = 0.0739 # mm/hr
    # rain cloud height = 3km
    rain_attenuation = 0.0000443 * math.pow(np.random.normal(rain_mean,rain_std), 1.0185) * 3

    # Cloud attenuation
    # A = columnar water content * attenuation coefficient *
    cloud_attenuation = 0.5 * 0.01 * 9.32

    return path_loss + rain_attenuation + cloud_attenuation


# Calculate the antenna gain

def antenna_gain(antenna_diameter, efficiency, frequency_band):
    gain_note = efficiency * ( math.pi * antenna_diameter * frequency_band/ 3e8)**2
    return 10 * math.log10(gain_note)


# Calculate RX power from GEO satellite to a user
def geo_to_user_rx(altitude, distance):

    # GEO to user distance
    geo_to_user_distance, angle  = calculate_distance(sat_altitude=altitude, horizontal_distance=distance )

    # (TX_power * GEO antenna gain * RX antenna gain) / (PL + A_T)
    geo_rx_signal = 10 * math.log10(geo_transmit_power) + antenna_gain(antenna_diameter=geo_antenna_diameter,
                                                                              efficiency=geo_antenna_efficiency,
                                                                              frequency_band=frequency) \
                    + antenna_gain(antenna_diameter=user_antenna_diameter, efficiency=user_antenna_efficiency,
                                   frequency_band=frequency) \
                    - calculate_path_loss(distance=geo_to_user_distance, frequency_band=frequency, angle=angle)

    return 10**(geo_rx_signal/10)


# Calculate RX power from LEO satellite to a user
def leo_to_user_rx(altitude, distance):

    # LEO to user distance
    leo_to_user_distance, angle = calculate_distance(sat_altitude = altitude, horizontal_distance=distance)

    # leo rx power at the user
    # (TX_power * LEO antenna gain * RX antenna gain) / (PL + A_T)
    leo_rx_signal = 10 * math.log10(leo_transmit_power) + antenna_gain(antenna_diameter=leo_antenna_diameter,
                                                                              efficiency=leo_antenna_efficiency,
                                                                              frequency_band=frequency) \
                    + antenna_gain(antenna_diameter=user_antenna_diameter, efficiency=user_antenna_efficiency,
                                   frequency_band=frequency) \
                    - calculate_path_loss(distance=leo_to_user_distance, frequency_band=frequency, angle= angle)
    # print(leo_transmit_power)

    return 10**(leo_rx_signal/10)


def leo_to_geo_user_rx(altitude, distance, bandwidth_ratio):

    # LEO to user distance
    leo_to_user_distance, angle = calculate_distance(sat_altitude = altitude, horizontal_distance=distance)

    # leo rx power at the user
    # (TX_power * LEO antenna gain * RX antenna gain) / (PL + A_T)
    leo_rx_signal = 10 * math.log10(leo_transmit_power) + antenna_gain(antenna_diameter=leo_antenna_diameter,
                                                                              efficiency=leo_antenna_efficiency,
                                                                              frequency_band=frequency) \
                    + antenna_gain(antenna_diameter=user_antenna_diameter, efficiency=user_antenna_efficiency,
                                   frequency_band=frequency) \
                    - calculate_path_loss(distance=leo_to_user_distance, frequency_band=frequency, angle= angle) \
                    + 10 * math.log10(1/bandwidth_ratio)
    # print(leo_transmit_power)

    return 10**(leo_rx_signal/10)

