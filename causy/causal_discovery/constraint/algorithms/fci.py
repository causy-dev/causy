from typing import Union

from causy.graph import Node
from causy.interfaces import ExtensionInterface


class InducingPathExtension(ExtensionInterface):
    class GraphAccessMixin:
        def inducing_path_exists(
            self, u: Union[Node, str], v: Union[Node, str]
        ) -> bool:
            """
            Check if an inducing path from u to v exists.
            An inducing path from u to v is a directed reference from u to v on which all mediators are colliders.
            :param u: node u
            :param v: node v
            :return: True if an inducing path exists, False otherwise
            """

            if isinstance(u, Node):
                u = u.id
            if isinstance(v, Node):
                v = v.id

            if not self.directed_path_exists(u, v):
                return False
            for path in self.directed_paths(u, v):
                for i in range(1, len(path) - 1):
                    r, w = path[i]
                    if not self.bidirected_edge_exists(r, w):
                        # TODO: check if this is correct (@sof)
                        return True
            return False
