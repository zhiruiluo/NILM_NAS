def power_config(data, threshold):
    data = data.dropna()
    avg = data.mean()
    if avg > threshold:
        return 1
    else:
        return 0