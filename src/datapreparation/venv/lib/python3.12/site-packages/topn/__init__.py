# flake8: noqa
import sys

if sys.version_info[0] >= 3:
    from topn.awesome_topn import awesome_topn, awesome_hstack_topn, awesome_hstack
else:
    from awesome_topn import awesome_topn, awesome_hstack_topn, awesome_hstack