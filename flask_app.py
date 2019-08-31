from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from matplotlib import style
import io
import base64
import json
import requests
import pandas as pd
import numpy as np
from mpltools import layout

style.use('dark_background')

app = Flask(__name__)


def make_pie(sizes, text, labels, ax):
    ax.axis('equal')
    width = 0.35
    kwargs = dict(startangle=180, autopct='%1.1f%%')
    outside, _, autotexts = ax.pie(sizes, radius=1.25, pctdistance=1 - 2 * width / 3,
                                   labels=None, **kwargs)
    for autotext in autotexts:
        autotext.set_color('white')

    plt.setp(outside, width=width * 1.5, edgecolor='white')

    kwargs = dict(size=20, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **kwargs, color='slategray')


def make_bar(ind, data, width, title, labels, ax):
    ax.bar(ind, data[0], width)
    for j, v in enumerate(data[0]):
        ax.text(j - .125, v - v / 2, str(int(round(v))), color='white', fontweight='bold')

    bottom = data[0]
    for i in range(1, len(data)):
        ax.bar(ind, data[i], width, bottom=bottom)
        if i < len(data) - 1:
            bottom += data[i]

        for j, v in enumerate(bottom.tolist()):
            ax.text(j - .125, v + data[i][j] / 2, str(int(round(data[i][j]))), color='white', fontweight='bold')

    ax.set_xticks(ind, minor=False)
    ax.set_xticklabels(labels)
    ax.set_title(title)


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
    # here we want to get the value of user (i.e. ?user=some-value)
    brand = request.args.get('brand')
    if not brand:
        return 'Missing brand'

    with open('input.json') as json_file:
        data = json.load(json_file)

    headers = dict()
    headers['Authorization'] = 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE1NjU2Mzc1NjksImV4cCI6MTU5NzE3MzU2OSwiaXNzIjoiTUtURyIsInVzZXJfaWQiOiI1ZDFiOGEwZjNlZjAyOTRmNGZkNjUyM2YiLCJlbWFpbCI6ImNodW4ud3VAbWt0Zy5jb20iLCJyb2xlX2lkIjoiNWFlMGVkZjBiZDg3MmU4NzNlNjk4YjAyIiwic2NvcGVzIjp7InBob3RvcG9ydGFsIjp7ImFjY2VzcyI6IkFkbWluIn19fQ.VOg35B3BmvLc14Q7m9lYXt9UIOefz5oHeVrMkQ-T3Ww'
    headers['Content-Type'] = 'application/json'

    # endpoint = 'http://127.0.0.1:5000/solver/solve'
    endpoint = 'https://martini.mktg.run/solver/solve'

    resp = requests.post(endpoint, headers=headers, json=data[brand.lower()])

    result = resp.json()['summary']

    calc_combs = ['brand-channel', 'channel-region', 'channel-month']
    solve = ['total_reach', 'total_cost', 'total_event']

    graph_url = []
    for calc in calc_combs:
        print(calc)
        output = pd.DataFrame(result[calc])

        if calc == 'brand-channel':
            labels = output['channel'].unique()

            fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharex='col')
            x = 0
            for s in solve:
                total = s + '\n' + f'{round(sum(output[s])):,}'
                make_pie(output[s], total, labels, ax[x])
                x += 1

            ax[0].legend(loc=2, labels=labels)

        else:
            index = calc.split('-')
            legend_index = index[0]
            merge_index = index[0]
            for i in index:
                if len(output[i].unique()) == 2:
                    legend = output[i].unique()
                    legend_index = i
                else:
                    labels = output[i].unique()
                    merge_index = i

            fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex='col')

            ind = np.arange(len(labels))
            width = .5

            x = 0
            for s in solve:
                # on premise
                on = output[output[legend_index] == legend[0]]
                off = output[output[legend_index] == legend[0]]

                onoff = pd.merge(on, off, on=[merge_index], how='outer')
                onoff.fillna(0, inplace=True)

                make_bar(ind, [onoff[s + '_x'], onoff[s + '_y']], width, s, labels, ax[x])
                x += 1

            ax[0].legend(loc=2, labels=legend)

        fig.suptitle(calc, fontsize=16)
        # Remove ticks on top and right sides of plot
        for a in ax.ravel():
            layout.cross_spines(ax=a)

        img = io.BytesIO()

        plt.savefig(img, format='png')
        img.seek(0)
        graph_url.append('data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode()))
        plt.close()

    #
    #
    #
    #
    # x1 = [0,1,2,3,4]
    # y1 = [10, 30, 40, 5, 50]
    # x2 = [0, 1, 2, 3, 4]
    # y2 = [10, 30, 40, 5, 50]
    # x3 = [0, 1, 2, 3, 4]
    # y3 = [10, 30, 40, 5, 50]
    #
    # graph1_url = build_graph(x1, y1)
    # graph2_url = build_graph(x2, y2)
    # graph3_url = build_graph(x3, y3)

    return render_template('graphs.html',
                           graph1=graph_url[0],
                           graph2=graph_url[1],
                           graph3=graph_url[2])


if __name__=='__main__':
    app.debug = True
    app.run()