"""Build an index of top-level symbols from Python modules and packages."""

import ast
import json
import logging
import os
import re
import sys
import hashlib
from importmagic.six.moves import cPickle as pickle
from contextlib import contextmanager
from distutils import sysconfig

from importmagic.util import parse_ast


LIB_LOCATIONS = sorted(set((
    (sysconfig.get_python_lib(standard_lib=True), 'S'),
    (sysconfig.get_python_lib(plat_specific=True), '3'),
    (sysconfig.get_python_lib(standard_lib=True, prefix=sys.prefix), 'S'),
    (sysconfig.get_python_lib(plat_specific=True, prefix=sys.prefix), '3'),
)), key=lambda l: -len(l[0]))

CACHE_LOCATION = os.path.join(os.path.expanduser('~'), '.cache', 'importmagic')


# Regex matching modules that we never attempt to index.
DEFAULT_BLACKLIST_RE = re.compile(r'\btest[s]?|test[s]?\b', re.I)
# Modules to treat as built-in.
#
# "os" is here mostly because it imports a whole bunch of aliases from other
# modules. The simplest way of dealing with that is just to import it and use
# vars() on it.
BUILTIN_MODULES = sys.builtin_module_names + ('os',)

LOCATION_BOOSTS = {
    '3': 1.2,
    'L': 1.5,
}


# TODO: Update scores based on import reference frequency.
# eg. if "sys.path" is referenced more than os.path, prefer it.


logger = logging.getLogger(__name__)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, SymbolIndex):
            d = o._tree.copy()
            d.update(('.' + name, getattr(o, name))
                     for name in SymbolIndex._SERIALIZED_ATTRIBUTES)
            return d
        return super(JSONEncoder, self).default(o)


def cached(method):
    def new_method(self, filename, *args, **kwds):
        if not self._cache_enabled or not filename:
            return method(self, filename, *args, **kwds)
        try:
            hash_base = (repr(filename) + str(os.path.getmtime(filename))).encode()
        except OSError:
            # file does not exit on disk
            return method(self, filename, *args, **kwds)
        sourcehash = hashlib.sha1(hash_base).hexdigest()
        cachefile = os.path.join(CACHE_LOCATION, sourcehash)
        try:
            if not os.path.isdir(CACHE_LOCATION):
                os.makedirs(CACHE_LOCATION)
            with open(cachefile, 'rb') as fp:
                result = pickle.load(fp)
        except Exception:
            result = method(self, filename, *args, **kwds)
            if result is not None:
                try:
                    with open(cachefile, 'wb') as fp:
                        pickle.dump(result, fp)
                except Exception:
                    logger.debug('failed to store cache file %s' %
                                 cachefile)
        return result
    return new_method


class SymbolIndex(object):
    PACKAGE_ALIASES = {
        # Give 'os.path' a score boost over posixpath and ntpath.
        'os.path': (os.path.__name__, 1.2),
        # Same with 'os', due to the heavy aliasing of other packages.
        'os': ('os', 1.2),
    }
    LOCATIONS = {
        'F': 'Future',
        '3': 'Third party',
        'S': 'System',
        'L': 'Local',
    }
    _PACKAGE_ALIASES = dict((v[0], (k, v[1])) for k, v in PACKAGE_ALIASES.items())
    _SERIALIZED_ATTRIBUTES = {'score': 1.0, 'location': '3'}

    def __init__(self, name=None, parent=None, score=1.0, location='3',
                 blacklist_re=None, cache_enabled=None):
        self._name = name
        self._tree = {}
        self._exports = {}
        self._parent = parent
        if blacklist_re:
            self._blacklist_re = blacklist_re
        elif parent:
            self._blacklist_re = parent._blacklist_re
        else:
            self._blacklist_re = DEFAULT_BLACKLIST_RE
        if cache_enabled is not None:
            self._cache_enabled = cache_enabled
        elif parent:
            self._cache_enabled = parent._cache_enabled
        else:
            self._cache_enabled = True
        self.score = score
        self.location = location
        if parent is None:
            self._merge_aliases()
            with self.enter('__future__', location='F'):
                pass
            with self.enter('__builtin__', location='S'):
                pass
        

    @classmethod
    def deserialize(self, file):
        def load(tree, data, parent_location):
            for key, value in data.items():
                if isinstance(value, dict):
                    score = value.pop('.score', 1.0)
                    location = value.pop('.location', parent_location)
                    with tree.enter(key, score=score, location=location) as subtree:
                        load(subtree, value, location)
                else:
                    assert isinstance(value, float), '%s expected to be float was %r' % (key, value)
                    tree.add(key, value)

        data = json.load(file)
        data.pop('.location', None)
        data.pop('.score', None)
        tree = SymbolIndex()
        load(tree, data, 'L')
        return tree

    @cached
    def _parse_source(self, filename, source=None):
        if source is None:
            with open(filename) as fd:
                source = fd.read()
        try:
            st = parse_ast(source, filename)
        except Exception as e:
            logger.debug('failed to parse %s: %s', filename, e)
            return
        visitor = SymbolVisitor()
        visitor.visit(st)
        return visitor.names, visitor.exports

    def index_source(self, filename, source):
        result = self._parse_source(filename, source)
        if result:
            self.update(*result)

    def index_file(self, module, filename):
        if self._blacklist_re.search(filename):
            return
        result = self._parse_source(filename)
        if result:
            with self.enter(module, location=self._determine_location_for(filename)) as subtree:
                subtree.update(*result)

    def index_path(self, root):
        """Index a path.

        :param root: Either a package directory, a .so or a .py module.
        """
        basename = os.path.basename(root)
        if basename.startswith('_') and os.path.splitext(basename)[0] != '__init__':
            return
        location = self._determine_location_for(root)
        if os.path.isfile(root):
            self._index_module(root, location)
        elif os.path.isdir(root) and os.path.exists(os.path.join(root, '__init__.py')):
            self._index_package(root, location)

    def _index_package(self, root, location):
        basename = os.path.basename(root)
        with self.enter(basename, location=location) as subtree:
            for filename in os.listdir(root):
                # exclude commonly found files directly
                if filename.endswith(('.pyc', '.pyo')):
                    continue
                subtree.index_path(os.path.join(root, filename))

    def _index_module(self, root, location):
        basename, ext = os.path.splitext(os.path.basename(root))
        if basename == '__init__':
            basename = None
        ext = ext.lower()
        import_path = '.'.join(filter(None, [self.path(), basename]))
        if import_path in BUILTIN_MODULES:
            return
        if ext == '.py':
            self.index_file(basename, root)
        elif ext in ('.dll', '.so'):
            self.index_builtin(import_path, root, location=location)

    @cached
    def _get_builtin_names(self, filename, module):
        try:
            module = __import__(module, fromlist=['.'])
        except Exception:
            logger.debug('failed to index builtin module %s', module)
            return []
        return [key for key in vars(module) if not key.startswith('_')]

    def index_builtin(self, module, filename, location):
        basemodule = module.rsplit('.', 1)[-1]
        if basemodule.startswith('_'):
            return
        names = self._get_builtin_names(filename, module)
        if names:
            with self.enter(basemodule, location=location) as subtree:
                for name in names:
                    subtree.add(name, 1.1)

    def build_index(self, paths):
        for builtin in BUILTIN_MODULES:
            self.index_builtin(builtin, '', location='S')
        for path in paths:
            # for the implicit "" entry in sys.path
            path = path or '.'
            if os.path.isdir(path):
                for filename in os.listdir(path):
                    filename = os.path.join(path, filename)
                    self.index_path(filename)

    def symbol_scores(self, symbol):
        """Find matches for symbol.

        :param symbol: A . separated symbol. eg. 'os.path.basename'
        :returns: A list of tuples of (score, package, reference|None),
            ordered by score from highest to lowest.
        """
        scores = []
        path = []

        # sys.path              sys path          ->    import sys
        # os.path.basename      os.path basename  ->    import os.path
        # basename              os.path basename   ->   from os.path import basename
        # path.basename         os.path basename   ->   from os import path
        def fixup(module, variable):
            prefix = module.split('.')
            if variable is not None:
                prefix.append(variable)
            seeking = symbol.split('.')
            new_module = []
            while prefix and seeking[0] != prefix[0]:
                new_module.append(prefix.pop(0))
            if new_module:
                module, variable = '.'.join(new_module), prefix[0]
            else:
                variable = None
            return module, variable

        def score_walk(scope, scale):
            sub_path, score = self._score_key(scope, full_key)
            if score > 0.1:
                try:
                    i = sub_path.index(None)
                    sub_path, from_symbol = sub_path[:i], '.'.join(sub_path[i + 1:])
                except ValueError:
                    from_symbol = None
                package_path = '.'.join(path + sub_path)
                package_path, from_symbol = fixup(package_path, from_symbol)
                scores.append((score * scale, package_path, from_symbol))

            for key, subscope in scope._tree.items():
                if type(subscope) is not float:
                    path.append(key)
                    score_walk(subscope, subscope.score * scale - 0.1)
                    path.pop()

        full_key = symbol.split('.')
        score_walk(self, 1.0)
        scores.sort(reverse=True)
        return scores

    def depth(self):
        depth = 0
        node = self
        while node._parent:
            depth += 1
            node = node._parent
        return depth

    def path(self):
        path = []
        node = self
        while node and node._name:
            path.append(node._name)
            node = node._parent
        return '.'.join(reversed(path))

    def find(self, path):
        """Return the node for a path, or None."""
        path = path.split('.')
        node = self
        while node._parent:
            node = node._parent
        for name in path:
            node = node._tree.get(name, None)
            if node is None or type(node) is float:
                return None
        return node

    def location_for(self, path):
        """Return the location code for a path."""
        path = path.split('.')
        node = self
        while node._parent:
            node = node._parent
        location = node.location
        for name in path:
            tree = node._tree.get(name, None)
            if tree is None or type(tree) is float:
                return location
            location = tree.location
        return location

    def add(self, name, score):
        current_score = self._tree.get(name, 0.0)
        if isinstance(current_score, float) and score > current_score:
            self._tree[name] = score

    def update(self, names, exports):
        for name, score in names.items():
            self.add(name, score)
        for name, score in exports.items():
            self._exports[name] = score

    @contextmanager
    def enter(self, name, location='L', score=1.0):
        if name is None:
            tree = self
        else:
            tree = self._tree.get(name)
            if not isinstance(tree, SymbolIndex):
                tree = self._tree[name] = SymbolIndex(name, self, score=score, location=location)
                if tree.path() in SymbolIndex._PACKAGE_ALIASES:
                    alias_path, _ = SymbolIndex._PACKAGE_ALIASES[tree.path()]
                    alias = self.find(alias_path)
                    alias._tree = tree._tree
        yield tree
        if tree._exports:
            # Delete unexported variables
            for key in set(tree._tree) - set(tree._exports):
                del tree._tree[key]

    def serialize(self, fd=None):
        if fd is None:
            return json.dumps(self, cls=JSONEncoder)
        return json.dump(self, fd, cls=JSONEncoder)

    def boost(self):
        return LOCATION_BOOSTS.get(self.location, 1.0)

    def __repr__(self):
        return '<%s:%r %r>' % (self.location, self.score, self._tree)

    def _merge_aliases(self):
        def create(node, alias, score):
            if not alias:
                return
            name = alias.pop(0)
            with node.enter(name, location='S', score=1.0 if alias else score) as index:
                create(index, alias, score)

        # Sort the aliases to always create 'os' before 'os.path'
        for alias in sorted(SymbolIndex._PACKAGE_ALIASES):
            package, score = SymbolIndex._PACKAGE_ALIASES[alias]
            create(self, package.split('.'), score)

    def _score_key(self, scope, key):
        if not key:
            return [], 0.0
        key_score = value = scope._tree.get(key[0], None)
        if value is None:
            return [], 0.0
        if type(value) is float:
            return [None, key[0]], key_score * scope.boost()
        else:
            path, score = self._score_key(value, key[1:])
            return [key[0]] + path, (score + value.score) * scope.boost()

    def _determine_location_for(self, path):
        for dir, location in LIB_LOCATIONS:
            if path.startswith(dir):
                return location
        return 'L'


class SymbolVisitor(ast.NodeVisitor):
    def __init__(self):
        self.exports = {}
        self.names = {}

    def visit_ImportFrom(self, node):
        for name in node.names:
            if name.name == '*' or name.name.startswith('_'):
                continue
            self.names[name.name] = 0.25

    def visit_Import(self, node):
        for name in node.names:
            if name.name.startswith('_'):
                continue
            self.names[name.name] = 0.25

    def visit_ClassDef(self, node):
        if not node.name.startswith('_'):
            self.names[node.name] = 1.1

    def visit_FunctionDef(self, node):
        if not node.name.startswith('_'):
            self.names[node.name] = 1.1

    def visit_Assign(self, node):
        # TODO: Handle __all__
        is_name = lambda n: isinstance(n, ast.Name)
        for name in filter(is_name, node.targets):
            if name.id == '__all__' and isinstance(node.value, ast.List):
                for subnode in node.value.elts:
                    if isinstance(subnode, ast.Str):
                        self.exports[subnode.s] = 1.2
            elif not name.id.startswith('_'):
                self.names[name.id] = 1.1

    def visit_If(self, node):
        # NOTE: In lieu of actually parsing if/else blocks at the top-level,
        # we'll just ignore them.
        pass


if __name__ == '__main__':
    # print ast.dump(ast.parse(open('pyautoimp.py').read(), 'pyautoimp.py'))
    tree = SymbolIndex()
    tree.build_index(sys.path)
    tree.serialize(sys.stdout)
