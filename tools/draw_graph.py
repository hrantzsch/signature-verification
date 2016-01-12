import sys
from graphviz import Source

with open(sys.argv[1]) as g:
    src = Source(g.read())

src.view()
