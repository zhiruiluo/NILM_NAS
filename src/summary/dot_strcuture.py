import graphviz

def get_node_attr(node):
    if node['class'] == 'conv':
        node_attr = dict(
            style='filled',
            fillcolor='green',
            aligh='left',
            fontsize='10',
            ranksep='0.1',
            height='0.2',
            fontname='monospace',
            shape='box',
            color='black'
        )
    else:
        node_attr = dict(
            # style='filled',
            aligh='left',
            fontsize='10',
            ranksep='0.1',
            height='0.2',
            fontname='monospace',
            shape='box',
            color='black'
        )
    return node_attr

def draw_graph(graph, dot = None):
    if dot is None:
        dot = graphviz.Digraph('round-table', format='png')
    
    
    
    for node in graph['nodes']:
        node_attr = get_node_attr(node)
        dot.node(node['id'],node['name'],**node_attr)
    
    for edge in graph['edges']:
        if isinstance(edge,tuple):
            dot.edge(edge[0], edge[1])
    
    return dot
    
    
def test():
    graph = {
        'nodes':[
            {'id': '1', 'class': 'input', 'name': 'input', 'dim': (3,3)},
            {'id': '2', 'class': 'conv','name': "3x3 Conv", 'dim': (3,10)}
        ],
        'edges':[
            ('1','2')
        ]
    }
    
    g = graphviz.Digraph('network_structure',  format='png',directory='doctest-output')
    g = draw_graph(graph, g)
    g.render().replace('\\','/')

