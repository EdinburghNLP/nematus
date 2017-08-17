#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict

class HypGraph(object):

    def __init__(self):
        self.nodes = defaultdict(str) # {id = label}
        self.edges = [] # (parent_node_id, child_node_id)
        self.costs = defaultdict(float) # {node_id = cost}
        self.word_probs = defaultdict(float) # {node_id = word_prob}

    def get_id(self, word, history):
        if history == []:
            return str(word)
        history = '-'.join([str(h) for h in reversed(history)])
        return '%s-%s' % (word, history)

    def get_ids(self, words):
        ids = []
        for i, w in enumerate(words):
            history = words[:i]
            ids.append(self.get_id(w, history))
        return ids

    def add(self, word, history, word_prob=None, cost=None):
        history_labels = [0] + history
        history_ids = self.get_ids(history_labels)
        word_label = word
        word_id = self.get_id(word_label, history_labels)
        # store
        self.nodes[word_id] = word_label
        self.edges.append((history_ids[-1], word_id))
        if word_prob != None:
            self.word_probs[word_id] = word_prob
        if cost != None:
            self.costs[word_id] = cost

class HypGraphRenderer(object):

    def __init__(self, hyp_graph):
        self.nodes = hyp_graph.nodes
        self.edges = hyp_graph.edges
        self.costs = hyp_graph.costs
        self.word_probs = hyp_graph.word_probs
        # constants
        self.BOS_SYMBOLS = ['0']
        self.EOS_SYMBOLS = ['<eos>']

    def _escape(self, word):
        replacements = {
            '<': '&lt;',
            '>': '&gt;',
        }
        for original, replacement in replacements.iteritems():
            word = word.replace(original, replacement)
        return word

    def _render(self, costs=False, word_probs=False, highlight_best=False):
        from pygraphviz import AGraph
        graph = AGraph(directed=True)
        for node_id, node_label in self.nodes.iteritems():
            attributes = self._node_attr(node_id, costs=costs, word_probs=word_probs)
            graph.add_node(node_id, **attributes)
        for (parent_node_id, child_node_id) in self.edges:
            graph.add_edge(parent_node_id, child_node_id)
        self.graph = graph
        if highlight_best:
            self._highlight_best()

    def _node_attr(self, node_id, costs=False, word_probs=False):
        word = self.nodes[node_id].decode('utf-8')
        word = self._escape(word)
        cost = self.costs[node_id] or ''
        prob = self.word_probs[node_id] or ''
        attr = {}
        if costs or word_probs:
            attr['shape'] = 'none'
            attr['margin'] = 0
            attr['label'] = '<<TABLE BORDER="0" CELLSPACING="0" CELLBORDER="1"><TR><TD>p=%.3f</TD><TD>%.3f</TD></TR><TR><TD COLSPAN="2">%s</TD></TR></TABLE>>' % (prob, cost, word)
        else:
            attr['label'] = word
        return attr

    def _highlight_best(self):
        best_hyp_bg_color = '#CDE9EC'
        best_hyp_cost = None
        best_hyp_leaf_node_id = None
        for node_id, label in self.nodes.iteritems():
            if label in self.EOS_SYMBOLS:
                if best_hyp_cost == None or self.costs[node_id] < best_hyp_cost:
                    best_hyp_leaf_node_id = node_id
                    best_hyp_cost = self.costs[node_id]
        if best_hyp_leaf_node_id:
            best_hyp_leaf_node = self.graph.get_node(best_hyp_leaf_node_id)
            current_node = best_hyp_leaf_node
            while current_node != []:
                current_node.attr['style'] = 'filled'
                current_node.attr['fillcolor'] = best_hyp_bg_color
                try:
                    current_node = self.graph.predecessors(current_node)[0]
                except IndexError:
                    break

    def wordify(self, word_dict):
        """
        Replace node labels (usually integers) with words, subwords, or
        characters.
        """
        for node_id, label in self.nodes.iteritems():
            self.nodes[node_id] = word_dict[label]

    def save(self, filepath, detailed=False, highlight_best=False):
        """
        Renders the graph as PNG image.

        @param filepath the taget file
        @param detailed whether to include word probabilities and
               hypothesis costs.
        @param highlight_best whether to highlight the best hypothesis.
        """
        costs = True if detailed else False
        word_probs = True if detailed else False
        self._render(costs=costs, word_probs=word_probs, highlight_best=highlight_best)
        self.graph.draw(filepath, prog="dot")
