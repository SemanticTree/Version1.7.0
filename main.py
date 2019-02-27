# MD-TPM -> Module Dependency to be cleared at time of module (Text Processing Module) integration.

# Imports
import numpy as np
import re
import scipy.cluster.vq as cluster_v
from itertools import islice, combinations
from collections import defaultdict, deque, OrderedDict
from collections import namedtuple
from pprint import pprint as print_
import spacy
from spacy.tokens import Token
import warnings

connection = namedtuple('connection', 'node weight')
Index = namedtuple('Index', 'pid, sid, wid')

nlp = spacy.load('en_core_web_lg')
spacy.load('/tmp/en_wiki', vocab=nlp.vocab)  # used for the time being

# spacy.load('en_vectors_web_lg', vocab=nlp.vocab)
warnings.filterwarnings("ignore")

# Global Variables
THRESHOLD = 0.7  # MD-TPM
TREE = None  # Final Tree Object
ROOT = None  # Final root of Tree
DOC = None  # Actual doc object for the function
SENT_RANGE = None
WORD_RANGE = None
SECTION_JOIN_THRESHHOLD = 0.95
NODE_TAGS = ['NOUN', 'ADJ']
PAIR_MAX_CONNECTIONS = 20


# Indexing of Tokens to a private system.
def set_index():
    """No-param function. Sets index <custom variable> for each token.
    Uses named tuple "Index" of the format(pid, sid, wid)
    Each successive paragraph, sentence, word recieves an incrementing ID

    Returns: None

    Note:
    Sets Global Variable <SENT_RANGE, WORD_RANGE, DOC>
    """
    global SENT_RANGE, WORD_RANGE, DOC
    SENT_RANGE = len(list(DOC.sents))
    WORD_RANGE = len(list(DOC))
    spacy.tokens.Token.set_extension('index', default=None, force=True)
    pc, sc, wc = 0, 0, 0
    for t in DOC:
        if t.text is '\n':
            pc += 1
        if t.is_sent_start:
            sc += 1
        if not t.is_punct and t.text is not '\n':
            t._.index = Index(pc, sc, wc)
            wc += 1


#  Frequency Counts and Instance List and unique tokens
def set_extentions(doc):
    """No-param function. Sets 'frequency' and 'instance_list' variable for each token.
    The frequency is calculated by lemma_.lower() word of the noun phrase.
    And lemma_.lower() is used to add instance to instance list.

    Returns: None
    """

    freq_count = defaultdict(int)
    instance_list = defaultdict(list)
    for t in doc:
        freq_count[t.lemma_.lower()] += 1
        instance_list[t.lemma_.lower()].append((t))

    def get_freq(t): return freq_count[t.lemma_.lower()]

    def get_instance_list(t): return instance_list[t.lemma_.lower()]

    spacy.tokens.Token.set_extension('frequency', getter=get_freq, force=True)
    spacy.tokens.Token.set_extension('instance_list', getter=get_instance_list, force=True)
    return doc


def make_unique(tokens):
    uniq = OrderedDict()
    for t in tokens:
        uniq.setdefault(t.lemma_.lower(), t)
    return list(uniq.values())

# Vector Functions
# Euclidean Distance Function


def euclidean_distance(x: np.ndarray, y: np.ndarray):
    """
    Calculate Euclidean distance of power 2

    Keyword Arguments:
    x -- First vector of dimension n
    y -- Second vector of dimension n
    """
    return np.linalg.norm(x - y)


# Similarity between vectors
def cosine_similarity(x: np.ndarray, y: np.ndarray):
    """
    Calculates Cosine Similarity : cos = A.B / |A||B|
    Answer value ranges from -1(Perfectly Opposite) to 1(perfectly Similar).
    Value of 0 translates to no similarity.

    Keyword Arguments:
    x -- First Vector of dimension n
    y -- Second Vector of dimension n
    """
    x_val, y_val = x.copy(), y.copy()
    assert x_val.shape == y_val.shape
    mod_x = (x_val ** 2).sum() ** 0.5
    mod_y = (y_val ** 2).sum() ** 0.5
    cos = x.dot(y) / (mod_x * mod_y)
    assert cos is not np.nan
    return cos


# Structural Distance Functions
# Paragraph Distance
# Implementing Static Calculation through : e^(-4x)
# Later versions may include dynamic calculations of z-score by getting all pairs of p_distance to produce more subtle(smooth) values. : a = (x-mean)/ std; z-score = scipy.stats.norm.cdf(a)


# v1.3 New Distance Function ::
def term_distance(term_a, term_b):
    """Calculates Term Distance using word sub-index by the formula : e^(x/r)
    r = word_range / sent_range
    """
    global DOC, SENT_RANGE, WORD_RANGE

    def min_dist_between_two_set(a, b):
        i, j = 0, 0
        min_diff = 100
        while i < len(a) and j < len(b):
            if min_diff > abs(a[i] - b[j]):
                x, y, min_diff = i, j, abs(a[i] - b[j])
            i, j = (i + 1, j) if a[i] < b[j] else (i, j + 1)
        return min_diff

    assert term_a._.instance_list is not []
    assert term_b._.instance_list is not []

    ax = [x._.index.wid for x in term_a._.instance_list]
    bx = [x._.index.wid for x in term_b._.instance_list]
    min_dist = min_dist_between_two_set(ax, bx)
    value = np.e ** -(min_dist * SENT_RANGE / WORD_RANGE)
    return value


# Frequency Sum MD-TMP 1.1
def freq_sum(a, b):
    """Get frequency Sum of the tokens a and b

    Keywords:
    a - 1st token object
    b - 2nd token object
    """
    summation = a._.frequency + b._.frequency
    return summation


# Compilation Functions

# Text to Pairs funtion.
def make_pairs(processed_text):
    """
    Accepts processed_text and outputs pair list as tuples.
    """
    global nlp, SECTION_JOIN_THRESHHOLD, NODE_TAGS, PAIR_MAX_CONNECTIONS, DOC

    DOC = nlp(processed_text)
    DOC = set_extentions(DOC)

    # 1. Make Sections - Based in new Lines
    index = [0] + [x.i for x in DOC if "\n" in x.text]
    sections = [DOC[i:j] for i, j in zip(index[:-1], index[1:])]

    # 2. Merge Sections
    logical_sections = []
    a = sections.pop(0)
    sec_start, sec_end = a.start, a.end
    while(sections):
        b = sections.pop(0)
        if a.similarity(b) > SECTION_JOIN_THRESHHOLD:
            sec_end = b.end
        else:
            logical_sections.append((sec_start, sec_end))
            a = b
            sec_start, sec_end = a.start, a.end
    logical_sections.append((sec_start, sec_end))

    # 3. Identify Nodes
    Major, Minor = [], []
    for i, j in logical_sections:
        sec = DOC[i:j]
        nodes = [ele for ele in sec if ele.pos_ in NODE_TAGS]
        selected_nodes = make_unique(nodes)
        nodes = np.array(selected_nodes[:])
        pmc = len(nodes) if PAIR_MAX_CONNECTIONS > len(nodes) else PAIR_MAX_CONNECTIONS
        simi_matrix = np.array([[x.similarity(y) for y in nodes] for x in nodes])
        sorted_ranks = np.fliplr(simi_matrix.argsort(axis=1))
        pair_list = zip(nodes, nodes[sorted_ranks[:, 1:pmc]])
        pairs = np.array([(a, b) for a, ls in pair_list for b in ls])
        Major.extend(list(pairs))
        top_nodes = nodes[simi_matrix.sum(axis=1).argsort()][:5]
        Minor.extend(top_nodes)

    selected_nodes = make_unique(Minor)
    nodes = np.array(selected_nodes[:])
    pmc = len(nodes) if PAIR_MAX_CONNECTIONS > len(nodes) else PAIR_MAX_CONNECTIONS
    simi_matrix = np.array([[x.similarity(y) for y in nodes] for x in nodes])
    sorted_ranks = np.fliplr(simi_matrix.argsort(axis=1))
    pair_list = zip(nodes, nodes[sorted_ranks[:, 1:pmc]])
    pairs = np.array([(a, b) for a, ls in pair_list for b in ls])
    Major.extend(list(pairs))
    unique_pairs = list({''.join(sorted([p[0].text, p[1].text])): tuple(p) for p in Major}.values())
    print(f'Total Pairs created from text : {len(unique_pairs)}')
    return unique_pairs


# Assign values - Compilation Functin - Key Pipeline Function
def assign_values(edge_list, weight_matrix=None):
    """
    Returns edge and value pair : ((node1, node2), edge_weight)

    Keyword:
    concept_list -  a list of unique concepts objects with properties: vector, p_id, s_id, w_id
    weight_matrix - weights for the various attributes.
    """
    global FREQ_COUNTS
    FREQ_COUNTS = None
    gathered_value = []

    for a, b in edge_list:
        cs = cosine_similarity(a.vector, b.vector)

        # Make a better fuction to get i values in future versions
        wd = term_distance(a, b)

        fs = freq_sum(a, b)
        arr = np.array([cs, wd, fs])
        gathered_value.append(arr)

    compiled = np.array(gathered_value)
    nrm = (compiled - compiled.min(axis=0)) / compiled.ptp(axis=0)

    w_mat = np.ones(3) / 3 if weight_matrix is None else weight_matrix
    w_normalised = nrm * w_mat

    total_nrm = w_normalised.sum(axis=1)
    print(f'Weight_values :::\n     Max values: {total_nrm.max()}\n     Min value: {total_nrm.min()}')

    pair = list(zip(edge_list, total_nrm))
    return pair


# Make thresholded graph
def make_graph(edge_list, threshold=0.0, max_connections=10):
    """Return 2 way graph from edge_list based on threshold"""
    graph = defaultdict(list)
    edge_list.sort(reverse=True, key=lambda x: x[1])
    for nodes, weight in edge_list:
        a, b = nodes
        if weight > threshold:
            if len(graph[a]) < max_connections:
                graph[a].append(connection(b, weight))
            if len(graph[b]) < max_connections:
                graph[b].append(connection(a, weight))
    print(f'Total graph nodes       : {len(graph.keys())}')
    print(f'Total graph connections : {sum(map(len, graph.values()))}')
    return graph


# Tree Generation Algorithm - Key Pipeline Function
def make_tree(graph):
    """
    Prepares a tree object from a graph based on edge strength. Determines the central node on its own.

    Keyword:
    graph -- A graph object(dict) containing list of connections as its value. E.g.
    { sapcy.token:node : [connection(node={spacy.token:node1}, weight={float:value}),..], ... }
    """
    tree = defaultdict(list)
    available = set(graph.keys())
    active = set()
    leaves = set()

    def _make_edge(parent, child, weight):
        child.set_extension('edge_strength', default=None, force=True)
        child._.edge_strength = weight
        child.set_extension('relation_to_parent', default=None, force=True)
        child._.relation_to_parent = get_relation(parent, child)[1]
        tree[parent].append(child)

    def get_max_from_available():
        return max(available, key=lambda x: x._.frequency)

    def get_max_from_active():
        return max(active, key=lambda x: graph[x][0].weight)

    root = get_max_from_available()
    available.remove(root)
    active.add(root)
    while(available):

        parent = get_max_from_active() if active else get_max_from_available()
        active.discard(parent)

        if not graph[parent]:
            leaves.add(parent)
            available.remove(parent)
            continue

        child, weight = graph[parent].pop(0)

        if child in available or child in leaves:  # danger
            _make_edge(parent, child, weight)
            available.remove(child)

            if graph[child]:
                active.add(child)

        if graph[parent]:
            active.add(parent)

    return tree, root


def get_relation(token1, token2):
    output = [(token1, DOC[i], token2) for i in range(token1.i, token2.i) if DOC[i].pos_ == "VERB"]
    if not output:
        output = (token1, 'has', token2)
    else:
        rels = [x[1] for x in output]
        rel2 = [x for x in rels if x.text not in ["is", "has", "have", "had", "was", "will"]]
        output = (token1, rels[int(len(rels) / 2)].text, token2)
    return output

# Dict object to Standard Dict - Key Pipeline Function


def make_a_node_dict(node):
    global TREE, ROOT
    node_dict = {}
    node_dict["title"] = node.text
    node_dict["i"] = node.sent.start_char  # MD-TPM 5.0
    node_dict["j"] = node.sent.end_char    # MD-TPM 5.0
    node_dict["relation_to_parent"] = node._.relation_to_parent  # MD-TPM 6.0
    node_dict["relation_strength"] = node._.edge_strength
    node_dict['is_central'] = node.text == ROOT.text
    node_dict["children"] = [ele for ele in map(make_a_node_dict, TREE[node])]
    return node_dict


# Transform StandardDict form to cytoscape form - Key Pipeline Function
def _transform_data(data: dict):
    """Accepts a data dictionary of standard format {Heirarchial format} and returns a
    format suitable for cytoscape.js"""
    new_dict = {'elements': []}
    elements = new_dict['elements']
    children, title = 'children', 'title'
    # r_st, rtp = 'relation_strength', 'relation_to_parent'

    def get_id():
        i = 1
        while True:
            yield 'edge' + str(i)
            i += 1

    id_generator = get_id()

    def add_node(node):
        elements.append({
            'data': {
                'id': node[title],
                'title': node[title],
                'has_child': node[children] == [],
                'i': node["i"],
                'j': node["j"],
                'is_central': node['is_central'],
            }
        })
        if node[children]:
            for a in node[children]:
                add_node(a)

        return

    def add_edge(node, parent):
        if node['relation_to_parent'] is not '-':
            if parent is not '-':
                elements.append({
                    'data': {
                        'id': next(id_generator),
                        'source': parent,
                        'target': node[title],
                        'title': node['relation_to_parent'],
                        'weight': node['relation_strength'],
                    }
                })
        if node[children]:
            for a in node[children]:
                add_edge(a, node[title])

        return

    new_dict = {'elements': []}
    elements = new_dict['elements']
    id_generator = get_id()
    add_node(data)
    add_edge(data, '-')
    return new_dict


def generate_structured_data_from_text(text, threshold_value=0.75,
                                       max_connections_value=5):
    """Returns cytoscape compatible json structure"""

    print("*" * 40)
    global DOC, TREE, ROOT, SENT_RANGE, WORD_RANGE

    # Add new line to the end for startuctural purposes and replace multiple new lines with single new line.
    processed_text = re.sub('(\\n)+', '\\n', text + '\n')
    pairs = make_pairs(processed_text)
    set_index()  # only when DOC object is set
    weight_matrix = np.array([0.3, 0.5, 0.2])  # cs, wd, fr

    # Skip from here for Module Integration
    a = assign_values(pairs, weight_matrix=weight_matrix, )
    g = make_graph(a, threshold=threshold_value, max_connections=max_connections_value)
    TREE, ROOT = make_tree(g)

    print(f'Root Node               : {ROOT}')
    print("Final Tree Created      :\n" + "*" * 40)

    standard_dict = make_a_node_dict(ROOT)
    cytoscape_dict = _transform_data(standard_dict)
    return cytoscape_dict


if __name__ == "__main__":
    sample_text = """
    An essay is, generally, a piece of writing that gives the author's own argument — but the definition is vague, overlapping with those of a paper, an article, a pamphlet, and a short story. Essays have traditionally been sub-classified as formal and informal. Formal essays are characterized by "serious purpose, dignity, logical organization, length," whereas the informal essay is characterized by "the personal element (self-revelation, individual tastes and experiences, confidential manner), humor, graceful style, rambling structure, unconventionality or novelty of theme," etc.[1]
    """

    print_(generate_structured_data_from_text(sample_text))
