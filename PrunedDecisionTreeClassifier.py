from sklearn import tree
from sklearn.tree._tree import TREE_LEAF
import pydotplus


class PrunedDecisionTreeClassifier(tree.DecisionTreeClassifier):
    
    def __init__(self,
                 pruning_threshold=20,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort=False):

        super(PrunedDecisionTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            presort=presort)

        self.pruning_threshold = pruning_threshold

    def prune_index(self, t, index=0):
        if t.value[index].min() < self.pruning_threshold:
            t.children_left[index] = TREE_LEAF
            t.children_right[index] = TREE_LEAF

        if t.children_left[index] != TREE_LEAF:
            self.prune_index(t, t.children_left[index])
            self.prune_index(t, t.children_right[index])

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        fit = super(PrunedDecisionTreeClassifier, self).fit(X, y, sample_weight, check_input, X_idx_sorted)
        self.prune_index(self.tree_)

        return fit


def save_tree(clf, out_dir):
    dot_data = tree.export_graphviz(clf)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(out_dir + '/tree.png')

