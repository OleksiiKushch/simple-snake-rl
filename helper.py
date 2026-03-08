import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython import display

plt.ion()

def plot(scores, mean_scores, total_games: int | None = None):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if total_games is not None:
        start_game = max(1, int(total_games) - len(scores) + 1)
        x = list(range(start_game, start_game + len(scores)))
        plt.plot(x, scores)
        plt.plot(x, mean_scores)
    else:
        plt.plot(scores)
        plt.plot(mean_scores)

    plt.ylim(ymin=0)

    if scores:
        last_x = (x[-1] if total_games is not None else (len(scores) - 1))
        plt.text(last_x, scores[-1], str(scores[-1]))
        plt.text(last_x, mean_scores[-1], str(mean_scores[-1]))

    plt.show(block=False)
    plt.pause(.1)