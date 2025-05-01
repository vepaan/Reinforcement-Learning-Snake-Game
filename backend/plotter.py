import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots()
fig.show()

def plot(scores, mean_scores):
    ax.clear()
    ax.plot(scores, label='score')
    ax.plot(mean_scores, label='mean')
    ax.set_title('Training…')
    ax.set_xlabel('No. of games')
    ax.set_ylabel('Score')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left')

    ax.text(len(scores)-1, scores[-1], str(scores[-1]))
    ax.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)
