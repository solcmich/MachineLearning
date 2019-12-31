# Load libraries

import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
import Input


class DecisionTree:

    def __init__(self, data):
        self.dct = dict()
        self.res = dict()
        self.res_s = list()
        self.res_s_i = list()
        self.sets = list()
        self.feature_cols = ['handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
                        'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools',
                        'nti-satellite-test-ban',
                        'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
                        'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports',
                        'export-administration-act-south-africa']
        self.data = data

    def build(self, data_path):
        col_names = ['handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
                     'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'nti-satellite-test-ban',
                     'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
                     'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports',
                     'export-administration-act-south-africa', 'result']

        data = pd.read_csv(data_path, header=None, names=col_names)
        head = data.head()

        X = data[self.feature_cols]
        y = data['result']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        # Create Decision Tree classifer object
        clf1 = DecisionTreeClassifier()

        # Train Decision Tree Classifer
        clf = clf1.fit(X_train, y_train)
        dot_data = export_graphviz(clf, out_file=None,
                                   filled=True, rounded=True,
                                   special_characters=True,
                                   feature_names=self.feature_cols, class_names=['0', '1'])
        graph = pydotplus.graph_from_dot_data(dot_data)
        # Image(graph.create_png())

    def ParseRule(self, txt):
        a = txt.split(' ')
        s = set()
        prev = -1
        res = False
        for i in a:
            if i == 'IF':
                continue
            elif i == 'AND':
                continue
            elif i == 'THEN':
                res = True
                continue
            elif '!' in i:
                i = '-' + i[1:]

            n = int(i)
            if res:
                self.res[prev] = n
                self.sets.append(s)
                self.res_s.append(s)
                self.res_s_i.append(n)
                break

            if prev != -1 and self.dct.get(prev):
                self.dct[prev].add(n)
            elif prev != -1 and not self.dct.get(prev):
                self.dct[prev] = {n}
            prev = n

            s.add(prev)

    def ProcessRules(self):
        f = open('rules.txt', "r")
        conts = f.readlines()
        for x in conts:
            self.ParseRule(x)

    def Decide(self, arr):
        set_r = set()

        for i in range(len(arr)):
            if arr[i] == 0:
                set_r.add(i*-1)
            else:
                set_r.add(i)
        for s in self.sets:
            if s.issubset(set_r):
                sets = self.res_s
                results = self.res_s_i
                for i in range(len(sets)):
                    if sets[i] == s:
                        return results[i]

    def isInSub(self, subs, sub):
        for i in subs:
            if sub.issubset(i):
                return True
        return False

    def filter_res(self, res, sub):
        ret = set()
        for i in res:
            tmp = set(sub)
            tmp.add(i)
            if self.isInSub(self.sets, tmp):
                ret.add(i)
        return ret

    def accuracy(self, data):
        l = ([row[:-1] for row in data])
        r = ([row[-1] for row in data])
        cnt = 0
        for i in range(len(l)):
            res = self.Decide(l[i])
            if abs(r[i] - res) < .5:
                cnt = cnt + 1
        return cnt / len(l)

    def decide_dialog(self):
        prr = self.sets
        main_val = int(Input.ask_for_feature(self.feature_cols[3]))
        if main_val == 0:
            main_val = -3
        if main_val == 1:
            main_val = 3

        set_r = {main_val}

        res = self.dct[main_val]
        q = list()
        for x in res:
            if abs(x) in [abs(x) for x in set_r]:
                val = 0
            else:
                if x < 0:
                    x = x * -1
                print("WHY?")
                for i in set_r:
                    if i > 0:
                        print(self.feature_cols[i])
                    else:
                        print('NOT ' + self.feature_cols[-i] + '->', end=' ')
                print()
                val = int(Input.ask_for_feature(self.feature_cols[x]))
            if val == 0:
                set_t = set(set_r)
                set_t.add(-x)
                if self.isInSub(self.sets, set_t):
                    q.append(-x)
                    set_r.add(-x)
            if val == 1:
                set_t = set(set_r)
                set_t.add(x)
                if self.isInSub(self.sets, set_t):
                    q.append(x)
                    set_r.add(x)
        while len(q) > 0:
            curr = q[-1]
            res = self.dct.get(curr)
            if res is None:
                ret = self.res[curr]
                print(ret)
                print("HOW?")
                for i in set_r:
                    if i > 0:
                        print(self.feature_cols[i])
                    else:
                        print('NOT ' + self.feature_cols[-i])
                break
            res = self.filter_res(res, set_r)
            q.pop()
            for x in res:
                if abs(x) in [abs(x) for x in set_r]:
                    val = 0
                else:
                    if x < 0:
                        x = x * -1
                    print("WHY?")
                    for i in set_r:
                        if i > 0:
                            print(self.feature_cols[-i] + '->', end= ' ')
                        else:
                            print('NOT ' + self.feature_cols[-i] + '->', end= ' ')
                    print('?')
                    val = int(Input.ask_for_feature(self.feature_cols[x]))
                if val == 0:
                    set_t = set(set_r)
                    set_t.add(-x)
                    if self.isInSub(self.sets, set_t):
                        q.append(-x)
                        set_r.add(-x)
                if val == 1:
                    set_t = set(set_r)
                    set_t.add(x)
                    if self.isInSub(self.sets, set_t):
                        q.append(x)
                        set_r.add(x)
