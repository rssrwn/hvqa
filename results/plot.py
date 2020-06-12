import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


TIMING_LABELS = ["Detector", "Properties", "Tracker", "Relations", "Events", "QA"]
TRAINED_TIMINGS = [56.8, 17.8, 1.88, 39.6, 5.18, 5.1]
HARDCODED_TIMINGS = [56.58, 18.25, 1.84, 22.81, 256.78, 7.04]

TRAINED_ACCS = [99.7, 98.8, 75.0, 60.8, 47.1]
HARDCODED_ACCS = [100, 97.9, 52.4, 44.2, 36.2]
HARDCODED_EC_ACCS = [100, 99.1, 52.8, 43.5, 36.0]
X_NAMES = ["0", "N/A", "0.1", "0.2", "0.3"]
FONT_SIZE = 26

EXPLODE = 0.0
ANGLE = -90


def plot_timings(trained):
    fig, ax = plt.subplots()

    data = TRAINED_TIMINGS if trained else HARDCODED_TIMINGS

    explode = [EXPLODE] * 6

    ax.pie(data, labels=TIMING_LABELS, autopct="%.0f%%", explode=explode, startangle=ANGLE, textprops={"size":"larger"})
    ax.axis("equal")

    plt.show()


def plot_accs(model):
    if model == "trained":
        data = TRAINED_ACCS
    elif model == "hardcoded":
        data = HARDCODED_ACCS
    elif model == "hardcoded-ec":
        data = HARDCODED_EC_ACCS

    fig, ax = plt.subplots(figsize=(6,4.5))

    formatter = FuncFormatter(lambda x, pos: f"{x:.0f}%")

    x = range(len(data))
    barlist = ax.bar(x, data)
    barlist[1].set_color('tab:pink')
    ax.yaxis.set_major_formatter(formatter)
    plt.xticks(x, X_NAMES)
    plt.xlabel("Error probability")
    plt.ylabel("Accuracy")

    plt.show()


def main(graph):
    if graph == "trained-timings":
        plot_timings(True)
    elif graph == "hardcoded-timings":
        plot_timings(False)
    elif graph == "trained-accs":
        plot_accs("trained")
    elif graph == "hardcoded-accs":
        plot_accs("hardcoded")
    elif graph == "hardcoded-ec-accs":
        plot_accs("hardcoded-ec")
    else:
        print("Unrecognised graph to plot.")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plot", type=str)
    args = parser.parse_args()
    main(args.plot)
