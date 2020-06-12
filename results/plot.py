import argparse
import matplotlib.pyplot as plt


TIMING_LABELS = ["Detector", "Properties", "Tracker", "Relations", "Events", "QA"]
TRAINED_TIMINGS = [56.8, 17.8, 1.88, 39.6, 5.18, 5.1]
HARDCODED_TIMINGS = [56.58, 18.25, 1.84, 22.81, 256.78, 7.04]

EXPLODE = 0.0
ANGLE = -60


def plot_timings(trained):
    fig, ax = plt.subplots()

    data = TRAINED_TIMINGS if trained else HARDCODED_TIMINGS

    explode = [EXPLODE] * 6

    ax.pie(data, labels=TIMING_LABELS, autopct="%.0f%%", explode=explode, startangle=ANGLE)
    ax.axis("equal")

    plt.show()


def main(graph):
    if graph == "trained-timings":
        plot_timings(True)
    elif graph == "hardcoded-timings":
        plot_timings(False)
    else:
        print("Unrecognised graph to plot.")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plot", type=str)
    args = parser.parse_args()
    main(args.plot)
