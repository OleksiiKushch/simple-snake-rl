import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython import display

plt.ion()

def plot(scores, mean_scores, epsilons=None, total_games: int | None = None):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')

    fig = plt.gcf()
    ax1 = plt.gca()
    ax1.set_ylabel('Score')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    if total_games is not None:
        start_game = max(1, int(total_games) - len(scores) + 1)
        x = list(range(start_game, start_game + len(scores)))
        l1, = ax1.plot(x, scores, label='Score')
        l2, = ax1.plot(x, mean_scores, label='Mean Score')
    else:
        x = list(range(len(scores)))
        l1, = ax1.plot(x, scores, label='Score')
        l2, = ax1.plot(x, mean_scores, label='Mean Score')

    ax1.set_ylim(ymin=0)

    lines = [l1, l2]
    labels = ['Score', 'Mean Score']

    if epsilons and len(epsilons) == len(scores):
        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon (exploration %)', color='grey')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelcolor='grey')
        l3, = ax2.plot(x, epsilons, color='grey', linestyle='--', alpha=0.6, label='Epsilon')
        lines.append(l3)
        labels.append('Epsilon')
        if epsilons:
            ax2.text(x[-1], epsilons[-1], f'{epsilons[-1]:.3f}', color='grey', fontsize=8)

    if scores:
        ax1.text(x[-1], scores[-1], str(scores[-1]))
        ax1.text(x[-1], mean_scores[-1], str(mean_scores[-1]))

    ax1.legend(lines, labels, loc='upper left', fontsize=8)

    plt.show(block=False)
    plt.pause(.1)