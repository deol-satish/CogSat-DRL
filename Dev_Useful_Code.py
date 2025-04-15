import pandas as pd
import re
import numpy as np

import pandas as pd
import re
import numpy as np

def build_satellite_data(df, row_index=0):
    result = {'LEO': {}, 'GEO': {}}

    for col in df.columns:
        if col == 'Time':
            continue

        match = re.match(r'(LEO|GEO)(\d+)_(.+)', col)
        if not match:
            continue

        sat_type, sat_num_str, rest = match.groups()
        sat_num = int(sat_num_str)

        if sat_type not in result:
            result[sat_type] = {}
        if sat_num not in result[sat_type]:
            result[sat_type][sat_num] = {}

        value = df.at[row_index, col] if col in df.columns else np.nan

        # Handle lat/lon
        if rest == 'Lat':
            result[sat_type][sat_num]['lat'] = value
        elif rest == 'Lon':
            result[sat_type][sat_num]['lon'] = value
        else:
            metric_match = re.match(r'(.+?)_(SNR|RSSI)', rest)
            if metric_match:
                location, metric = metric_match.groups()
                if 'GS_metrics' not in result[sat_type][sat_num]:
                    result[sat_type][sat_num]['GS_metrics'] = {}
                if location not in result[sat_type][sat_num]['GS_metrics']:
                    result[sat_type][sat_num]['GS_metrics'][location] = {'SNR': np.nan, 'RSSI': np.nan}
                result[sat_type][sat_num]['GS_metrics'][location][metric] = value

    return result



import pandas as pd
import numpy as np

def parse_satellite_data(row):
    leo_data = {}
    geo_data = {}

    for col in row.index:
        if col.startswith("LEO"):
            parts = col.split('_')
            sat_id = int(parts[0].replace("LEO", ""))
            field = parts[1]
            
            if field in ["Lat", "Lon"]:
                if sat_id not in leo_data:
                    leo_data[sat_id] = {'lat': None, 'lon': None, 'GS_metrics': {}}
                leo_data[sat_id][field.lower()] = row[col]
            else:
                gs_location = field
                metric = parts[2]
                if sat_id not in leo_data:
                    leo_data[sat_id] = {'lat': None, 'lon': None, 'GS_metrics': {}}
                if gs_location not in leo_data[sat_id]['GS_metrics']:
                    leo_data[sat_id]['GS_metrics'][gs_location] = {}
                leo_data[sat_id]['GS_metrics'][gs_location][metric] = row[col]

        elif col.startswith("GEO"):
            parts = col.split('_')
            sat_id = int(parts[0].replace("GEO", ""))
            gs_location = parts[1]
            metric = parts[2]

            if sat_id not in geo_data:
                geo_data[sat_id] = {'GS_metrics': {}}
            if gs_location not in geo_data[sat_id]['GS_metrics']:
                geo_data[sat_id]['GS_metrics'][gs_location] = {}
            geo_data[sat_id]['GS_metrics'][gs_location][metric] = row[col]

    return {'LEO': leo_data, 'GEO': geo_data}




import pandas as pd
import re
import numpy as np

def build_all_satellite_data(df):
    results = []

    for i, row in df.iterrows():
        row_data = {'LEO': {}, 'GEO': {}}

        if 'Time' in df.columns:
            row_data['Time'] = row['Time']

        for col in df.columns:
            if col == 'Time':
                continue

            match = re.match(r'(LEO|GEO)(\d+)_(.+)', col)
            if not match:
                continue

            sat_type, sat_num_str, rest = match.groups()
            sat_num = int(sat_num_str)

            if sat_num not in row_data[sat_type]:
                row_data[sat_type][sat_num] = {}

            value = row[col]

            if rest == 'Lat':
                row_data[sat_type][sat_num]['lat'] = value
            elif rest == 'Lon':
                row_data[sat_type][sat_num]['lon'] = value
            else:
                metric_match = re.match(r'(.+?)_(SNR|RSSI)', rest)
                if metric_match:
                    location, metric = metric_match.groups()
                    if 'GS_metrics' not in row_data[sat_type][sat_num]:
                        row_data[sat_type][sat_num]['GS_metrics'] = {}
                    if location not in row_data[sat_type][sat_num]['GS_metrics']:
                        row_data[sat_type][sat_num]['GS_metrics'][location] = {'SNR': np.nan, 'RSSI': np.nan}
                    row_data[sat_type][sat_num]['GS_metrics'][location][metric] = value

        results.append(row_data)

    return results

