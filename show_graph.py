import os
import matplotlib.pyplot as plt

def show_graphs(metrics, num_epochs):
    os.makedirs('./loss_and_accuracy', exist_ok=True)
    graph_path = './loss_and_accuracy'

    for title, data in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(data) + 1), data)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.xlim(1, num_epochs)
        if 'Accuracy' in title:
            plt.ylim(0, 100)
            max_value = max(data)
            max_index = data.index(max_value)

            # 최대값에 텍스트 추가
            plt.text(max_index, max_value, f'Max: {max_value:.2f}', ha='center', color='red', va='bottom')

        plt.grid(True)
        plt.savefig(os.path.join(graph_path, f'{title.replace(" ", "_")}.png'))
        plt.close()
