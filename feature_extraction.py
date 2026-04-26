import numpy as np

def extract_features(gesture):
    if len(gesture) < 2:
        return [0]*15

    times = np.array([p["time"] for p in gesture])
    xs = np.array([p["x"] for p in gesture])
    ys = np.array([p["y"] for p in gesture])
    pressures = np.array([p["pressure"] for p in gesture])

    # CORE FEATURES
    duration = times[-1] - times[0]
    start_x, start_y = xs[0], ys[0]
    end_x, end_y = xs[-1], ys[-1]

    mean_pressure = np.mean(pressures)
    pressure_std = np.std(pressures)

    displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

    # DYNAMIC FEATURES
    speeds = []
    for i in range(1, len(gesture)):
        dt = times[i] - times[i-1]
        if dt == 0:
            continue
        dist = np.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2)
        speeds.append(dist / dt)

    speeds = np.array(speeds) if len(speeds) > 0 else np.array([0])

    mean_speed = np.mean(speeds)
    speed_std = np.std(speeds)

    slope = (end_y - start_y) / (end_x - start_x + 1e-6)

    deviations = []
    for i in range(len(xs)):
        num = abs((end_y - start_y)*xs[i] - (end_x - start_x)*ys[i] + end_x*start_y - end_y*start_x)
        den = np.sqrt((end_y - start_y)**2 + (end_x - start_x)**2 + 1e-6)
        deviations.append(num / den)

    mean_dev = np.mean(deviations)

    # MULTI-TOUCH FEATURES
    start_ifd = 0
    end_ifd = 0
    scale_factor = 1

    if "fingers" in gesture[0] and len(gesture[0]["fingers"]) == 2:
        f_start = gesture[0]["fingers"]
        f_end = gesture[-1]["fingers"]

        start_ifd = np.linalg.norm(np.array(f_start[0]) - np.array(f_start[1]))
        end_ifd = np.linalg.norm(np.array(f_end[0]) - np.array(f_end[1]))

        scale_factor = end_ifd / (start_ifd + 1e-6)

    return [
        duration, start_x, start_y, end_x, end_y,
        mean_pressure, pressure_std,
        displacement,
        mean_speed, speed_std,
        slope, mean_dev,
        start_ifd, end_ifd, scale_factor
    ]