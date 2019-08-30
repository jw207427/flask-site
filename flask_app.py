from flask import Flask, render_template
import matplotlib.pyplot as plt
import io
import base64


app = Flask(__name__)


def build_graph(x, y):
    img = io.BytesIO()
    plt.plot(x, y)
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)


@app.route('/graphs')
def graphs():
    x1 = [0,1,2,3,4]
    y1 = [10, 30, 40, 5, 50]
    x2 = [0, 1, 2, 3, 4]
    y2 = [10, 30, 40, 5, 50]
    x3 = [0, 1, 2, 3, 4]
    y3 = [10, 30, 40, 5, 50]

    graph1_url = build_graph(x1, y1)
    graph2_url = build_graph(x2, y2)
    graph3_url = build_graph(x3, y3)

    return render_template('graphs.html',
                           graph1=graph1_url,
                           graph2=graph2_url,
                           graph3=graph3_url)


if __name__=='__main__':
    app.debug = True
    app.run()