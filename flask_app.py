from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import io
import os
import base64
import json
import requests
import pandas as pd
import numpy as np
import seaborn as sns
from mpltools import layout
import calendar

mpl.rcParams['font.family'] = "Arial Rounded MT Bold"

sns.set_context("paper", font_scale=1.25)
# to change default color cycle
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.magma.colors)

style.use('dark_background')

app = Flask(__name__)


def make_pie(sizes, text, ax):
    ax.axis('equal')
    width = 0.35

    kwargs = dict(startangle=180, autopct='%1.1f%%')
    outside, _, autotexts = ax.pie(sizes, colors=['red', 'silver'], radius=1.25, pctdistance=1 - 2 * width / 3,
                                   labels=None, **kwargs)
    for autotext in autotexts:
        autotext.set_color('white')

    plt.setp(outside, width=width * 1.5, edgecolor='white')

    kwargs = dict(size=20, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **kwargs, color='white')


def make_bar(ind, data, width, title, labels, ax):
    bottom = np.zeros(len(ind))
    colors = ['red', 'gray']
    for i in range(len(data)):
        ax.bar(ind, data[i], width, bottom=bottom, align='center', color=colors[i])
        for j, v in enumerate(bottom.tolist()):
            ax.text(j, v + data[i][j] / 2, str(int(round(data[i][j]))), color='white', ha='center')
        bottom += data[i]

    title_kwargs = dict(size=20, va='bottom', fontweight='bold')
    label_kwargs = dict(va='top', fontweight='bold', rotation='vertical')
    ax.set_xticks(ind, minor=False)
    ax.set_xticklabels(labels, **label_kwargs)
    ax.set_title(title, **title_kwargs)


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

    this_folder = os.getcwd()

    with open(this_folder+'/input.json') as json_file:
        data = json.load(json_file)

    headers = dict()
    headers['Authorization'] = 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE1NjU2Mzc1NjksImV4cCI6MTU5NzE3MzU2OSwiaXNzIjoiTUtURyIsInVzZXJfaWQiOiI1ZDFiOGEwZjNlZjAyOTRmNGZkNjUyM2YiLCJlbWFpbCI6ImNodW4ud3VAbWt0Zy5jb20iLCJyb2xlX2lkIjoiNWFlMGVkZjBiZDg3MmU4NzNlNjk4YjAyIiwic2NvcGVzIjp7InBob3RvcG9ydGFsIjp7ImFjY2VzcyI6IkFkbWluIn19fQ.VOg35B3BmvLc14Q7m9lYXt9UIOefz5oHeVrMkQ-T3Ww'
    headers['Content-Type'] = 'application/json'

    # endpoint = 'http://127.0.0.1:5000/solver/solve'
    endpoint = 'https://martini.mktg.run/solver/solve'

    if brand.lower() not in data:
        return 'Cannot find this brand'

    resp = requests.post(endpoint, headers=headers, json=data[brand.lower()])

    result = resp.json()['summary']

    calc_combs = {'brand-channel':'Brand & Channel',
                  'channel-region': 'Channel & Region',
                  'channel-month': 'Channel & Month'}

    solve = ['total_reach', 'total_cost', 'total_event']

    graph_url = []
    title = []
    for calc in calc_combs:
        print(calc)
        output = pd.DataFrame(result[calc])

        if 'month' in list(output):
            output['month'] = output['month'].apply(lambda x: calendar.month_name[x])

        if calc == 'brand-channel':
            labels = output['channel'].unique()

            fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharex='col')
            x = 0
            for s in solve:
                total = s + '\n' + f'{int(round(sum(output[s]))):,}'
                make_pie(output[s], total, ax[x])
                x += 1

            ax[0].legend(loc=2, labels=labels)

        else:
            index = calc.split('-')
            legend_index = index[0]
            merge_index = index[0]
            for i in index:
                if len(output[i].unique()) <= 2:
                    legend = output[i].unique()
                    legend_index = i
                else:
                    labels = output[i].unique()
                    merge_index = i

            if len(labels) > 5:
                fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex='col')
            else:
                fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharex='col')

            ind = np.arange(len(labels))
            width = .5

            x = 0
            for s in solve:
                # on premise
                on = output[output[legend_index] == legend[0]]
                off = output[output[legend_index] != legend[0]]

                onoff = pd.merge(on, off, on=[merge_index], how='outer')
                onoff.fillna(0, inplace=True)

                make_bar(ind, [onoff[s + '_x'], onoff[s + '_y']], width, s, labels, ax[x])
                x += 1

            ax[0].legend(loc=2, labels=legend)

        # fig.suptitle(calc, fontsize=16)
        # Remove ticks on top and right sides of plot
        for a in ax.ravel():
            layout.cross_spines(ax=a)

        plt.tight_layout()

        img = io.BytesIO()

        plt.savefig(img, format='png')
        img.seek(0)
        graph_url.append('data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode()))
        title.append(calc_combs[calc])
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
                           title1=title[0],
                           title2=title[1],
                           title3=title[2],
                           graph1=graph_url[0],
                           graph2=graph_url[1],
                           graph3=graph_url[2])


@app.route('/')
def root():
    return 'Test site'


if __name__=='__main__':
    app.debug = True
    app.run()