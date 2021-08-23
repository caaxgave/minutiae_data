#FUNCTIONS

def extract_features(Values, angle=False):
    sorted_values = np.sort(Values)

    min_value = min(Values)
    max_value = max(Values)
    avrg_value = np.average(Values)
    variance_value = np.var(Values)
    median_value = np.median(Values)
    # std_value = np.std(Values)
    minmax_ratio = min_value / max_value

    if angle:
        avrg_value = degrees(phase(sum(rect(1, radians(d)) for d in Values) / len(Values)))

    second_min = sorted_values[1]
    second_max = sorted_values[len(Values) - 2]

    return [min_value, max_value, avrg_value, variance_value, median_value, minmax_ratio, second_min, second_max]


