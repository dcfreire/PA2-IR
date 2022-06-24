# Natural Language Toolkit: Internal utility functions
#
# Copyright (C) 2001-2022 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
#         Nitin Madnani <nmadnani@ets.org>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
# Modified by Daniel Carneiro Freire (daniel.carneiro.freire@gmail.com)


import re
import types
import os
import stat
from xml.etree import ElementTree


def raise_unorderable_types(ordering, a, b):
    raise TypeError(
        "unorderable types: %s() %s %s()"
        % (type(a).__name__, ordering, type(b).__name__)
    )

_STRING_START_RE = re.compile(r"[uU]?[rR]?(\"\"\"|\'\'\'|\"|\')")


def read_str(s, start_position):
    """
    If a Python string literal begins at the specified position in the
    given string, then return a tuple ``(val, end_position)``
    containing the value of the string literal and the position where
    it ends.  Otherwise, raise a ``ReadError``.

    :param s: A string that will be checked to see if within which a
        Python string literal exists.
    :type s: str

    :param start_position: The specified beginning position of the string ``s``
        to begin regex matching.
    :type start_position: int

    :return: A tuple containing the matched string literal evaluated as a
        string and the end position of the string literal.
    :rtype: tuple(str, int)

    :raise ReadError: If the ``_STRING_START_RE`` regex doesn't return a
        match in ``s`` at ``start_position``, i.e., open quote. If the
        ``_STRING_END_RE`` regex doesn't return a match in ``s`` at the
        end of the first match, i.e., close quote.
    :raise ValueError: If an invalid string (i.e., contains an invalid
        escape sequence) is passed into the ``eval``.

    :Example:

    >>> from nltk.internals import read_str
    >>> read_str('"Hello", World!', 0)
    ('Hello', 7)

    """
    # Read the open quote, and any modifiers.
    m = _STRING_START_RE.match(s, start_position)
    if not m:
        raise ReadError("open quote", start_position)
    quotemark = m.group(1)

    # Find the close quote.
    _STRING_END_RE = re.compile(r"\\|%s" % quotemark)
    position = m.end()
    while True:
        match = _STRING_END_RE.search(s, position)
        if not match:
            raise ReadError("close quote", position)
        if match.group(0) == "\\":
            position = match.end() + 1
        else:
            break

    # Process it, using eval.  Strings with invalid escape sequences
    # might raise ValueError.
    try:
        return eval(s[start_position : match.end()]), match.end()
    except ValueError as e:
        raise ReadError("valid escape sequence", start_position) from e

class ReadError(ValueError):
    """
    Exception raised by read_* functions when they fail.
    :param position: The index in the input string where an error occurred.
    :param expected: What was expected when an error occurred.
    """

    def __init__(self, expected, position):
        ValueError.__init__(self, expected, position)
        self.expected = expected
        self.position = position

    def __str__(self):
        return f"Expected {self.expected} at {self.position}"

class Counter:
    """
    A counter that auto-increments each time its value is read.
    """

    def __init__(self, initial_value=0):
        self._value = initial_value

    def get(self):
        self._value += 1
        return self._value

def overridden(method):
    """
    :return: True if ``method`` overrides some method with the same
        name in a base class.  This is typically used when defining
        abstract base classes or interfaces, to allow subclasses to define
        either of two related methods:

        >>> class EaterI:
        ...     '''Subclass must define eat() or batch_eat().'''
        ...     def eat(self, food):
        ...         if overridden(self.batch_eat):
        ...             return self.batch_eat([food])[0]
        ...         else:
        ...             raise NotImplementedError()
        ...     def batch_eat(self, foods):
        ...         return [self.eat(food) for food in foods]

    :type method: instance method
    """
    if isinstance(method, types.MethodType) and method.__self__.__class__ is not None:
        name = method.__name__
        funcs = [
            cls.__dict__[name]
            for cls in _mro(method.__self__.__class__)
            if name in cls.__dict__
        ]
        return len(funcs) > 1
    else:
        raise TypeError("Expected an instance method.")

def _mro(cls):
    """
    Return the method resolution order for ``cls`` -- i.e., a list
    containing ``cls`` and all its base classes, in the order in which
    they would be checked by ``getattr``.  For new-style classes, this
    is just cls.__mro__.  For classic classes, this can be obtained by
    a depth-first left-to-right traversal of ``__bases__``.
    """
    if isinstance(cls, type):
        return cls.__mro__
    else:
        mro = [cls]
        for base in cls.__bases__:
            mro.extend(_mro(base))
        return mro

class ElementWrapper:
    """
    A wrapper around ElementTree Element objects whose main purpose is
    to provide nicer __repr__ and __str__ methods.  In addition, any
    of the wrapped Element's methods that return other Element objects
    are overridden to wrap those values before returning them.

    This makes Elements more convenient to work with in
    interactive sessions and doctests, at the expense of some
    efficiency.
    """

    # Prevent double-wrapping:
    def __new__(cls, etree):
        """
        Create and return a wrapper around a given Element object.
        If ``etree`` is an ``ElementWrapper``, then ``etree`` is
        returned as-is.
        """
        if isinstance(etree, ElementWrapper):
            return etree
        else:
            return object.__new__(ElementWrapper)

    def __init__(self, etree):
        r"""
        Initialize a new Element wrapper for ``etree``.

        If ``etree`` is a string, then it will be converted to an
        Element object using ``ElementTree.fromstring()`` first:

            >>> ElementWrapper("<test></test>")
            <Element "<?xml version='1.0' encoding='utf8'?>\n<test />">

        """
        if isinstance(etree, str):
            etree = ElementTree.fromstring(etree)
        self.__dict__["_etree"] = etree

    def unwrap(self):
        """
        Return the Element object wrapped by this wrapper.
        """
        return self._etree

    ##////////////////////////////////////////////////////////////
    # { String Representation
    ##////////////////////////////////////////////////////////////

    def __repr__(self):
        s = ElementTree.tostring(self._etree, encoding="utf8").decode("utf8")
        if len(s) > 60:
            e = s.rfind("<")
            if (len(s) - e) > 30:
                e = -20
            s = f"{s[:30]}...{s[e:]}"
        return "<Element %r>" % s

    def __str__(self):
        """
        :return: the result of applying ``ElementTree.tostring()`` to
        the wrapped Element object.
        """
        return (
            ElementTree.tostring(self._etree, encoding="utf8").decode("utf8").rstrip()
        )

    ##////////////////////////////////////////////////////////////
    # { Element interface Delegation (pass-through)
    ##////////////////////////////////////////////////////////////

    def __getattr__(self, attrib):
        return getattr(self._etree, attrib)

    def __setattr__(self, attr, value):
        return setattr(self._etree, attr, value)

    def __delattr__(self, attr):
        return delattr(self._etree, attr)

    def __setitem__(self, index, element):
        self._etree[index] = element

    def __delitem__(self, index):
        del self._etree[index]

    def __setslice__(self, start, stop, elements):
        self._etree[start:stop] = elements

    def __delslice__(self, start, stop):
        del self._etree[start:stop]

    def __len__(self):
        return len(self._etree)

    ##////////////////////////////////////////////////////////////
    # { Element interface Delegation (wrap result)
    ##////////////////////////////////////////////////////////////

    def __getitem__(self, index):
        return ElementWrapper(self._etree[index])

    def __getslice__(self, start, stop):
        return [ElementWrapper(elt) for elt in self._etree[start:stop]]

    def getchildren(self):
        return [ElementWrapper(elt) for elt in self._etree]

    def getiterator(self, tag=None):
        return (ElementWrapper(elt) for elt in self._etree.getiterator(tag))

    def makeelement(self, tag, attrib):
        return ElementWrapper(self._etree.makeelement(tag, attrib))

    def find(self, path):
        elt = self._etree.find(path)
        if elt is None:
            return elt
        else:
            return ElementWrapper(elt)

    def findall(self, path):
        return [ElementWrapper(elt) for elt in self._etree.findall(path)]

def is_writable(path):
    # Ensure that it exists.
    if not os.path.exists(path):
        return False

    # If we're on a posix system, check its permissions.
    if hasattr(os, "getuid"):
        statdata = os.stat(path)
        perm = stat.S_IMODE(statdata.st_mode)
        # is it world-writable?
        if perm & 0o002:
            return True
        # do we own it?
        elif statdata.st_uid == os.getuid() and (perm & 0o200):
            return True
        # are we in a group that can write to it?
        elif (statdata.st_gid in [os.getgid()] + os.getgroups()) and (perm & 0o020):
            return True
        # otherwise, we can't write to it.
        else:
            return False

    # Otherwise, we'll assume it's writable.
    # [xx] should we do other checks on other platforms?
    return True