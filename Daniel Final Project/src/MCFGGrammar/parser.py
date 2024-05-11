from abc import ABC
from typing import Literal
from functools import lru_cache
from .grammar import *
from .trees import *

class Agenda:
    """
    This keeps track of all relevant rules of the grammar in a cache
    """
    def __init__(self, grammar: MCFGGrammar):
        self._cache = [rule for rule in grammar.rules() if rule.is_epsilon]

    def select_item(self):
        return self._cache.pop(0)

    def add_items(self, trigger: list[MCFGRule]):
        self._cache += trigger

    @property
    def is_empty(self):
        return not self._cache

MCFGBackPointer = tuple[str, SpanIndices]

class Chart(ABC):

    @property
    def parses(self):
        raise NotImplementedError

class ChartEntry(ABC):

    def __hash__(self) -> int:
        raise NotImplementedError

    @property
    def backpointers(self):
        raise NotImplementedError


class MCFGChartEntry(ChartEntry):
    """
    A chart entry for a MCFG chart

    Parameters
    ----------
    symbol
    backpointers

    Attributes
    ----------
    symbol
    backpointers
    """

    def __init__(self, symbol: MCFGRuleElementInstance, *backpointers: MCFGBackPointer):
        self._symbol = symbol
        self._backpointers = backpointers

    def to_tuple(self):
        return (self._symbol, self._backpointers)

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __eq__(self, other: 'MCFGChartEntry') -> bool:
        return self.to_tuple() == other.to_tuple()

    def __repr__(self) -> str:
        return str(self._symbol) + ' -> ' + ' '.join(
            f"{str(bp)}"
            for bp in self.backpointers
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def backpointers(self) -> tuple[MCFGBackPointer, ...]:
        return self._backpointers

class MCFGChart(Chart):
    """
    A chart for a MCFG parser

    Parameters
    ----------
    string

    Attributes
    ----------
    parses
    """

    def __init__(self, string): #input_size: int):
        self._input_size = len(string) #input_size
        self._string = string

        self._chart: dict[SpanIndices: set[MCFGChartEntry]] = dict() #{

    def __getitem__(self, index: SpanIndices) -> set[MCFGChartEntry]:
        if index not in self._chart:
            self._chart[index] = set({})

        return self._chart[index]

    def __setitem__(self, key: SpanIndices, item: set[MCFGChartEntry]):

        self._chart[key] = item

    def __iter__(self):
        return self._chart.__iter__()

    def _find_adjacent(self):
        candidates = []
        constituents = [elem for entry in self for elem in self[entry]]

        for item1 in constituents:
            elem1 = item1.symbol
            for item2 in constituents:
                elem2 = item2.symbol

                left_edges = [first[0] for first in list(elem1.string_spans) + list(elem2.string_spans)]
                right_edges = [last[-1] for last in list(elem1.string_spans) + list(elem2.string_spans)]

                if len([left_edge for left_edge in left_edges if left_edge not in right_edges]) == 1:
                    candidates.append((item1,item2))

        return candidates

    def add_item(self, trigger):
        new_triggers = set({})
        if trigger.is_epsilon:
            for i in range(self._input_size):
                if self._string[i] in trigger.left_side.unique_string_variables:
                    element = trigger.instantiate_left_side(MCFGRuleElementInstance(self._string[i], (i,i+1)))
                    self[element.string_spans].add(
                        MCFGChartEntry(
                            element,
                             (self._string[i], (element.string_spans,)))
                    ) # backpointer
                    new_triggers.add(element)

        else:

            if len(trigger.left_side.string_variables) > 1:

                pairs = [(item1, item2)
                        for i1 in self
                        for i2 in self
                        for item1 in self[i1]
                        for item2 in self[i2]]

            else:
                pairs = self._find_adjacent()

            for item1, item2 in pairs:
                if trigger._right_side_aligns((item1.symbol, item2.symbol)):
                    try:
                        element = trigger.instantiate_left_side(item1.symbol, item2.symbol)
                        self[element.string_spans].add(
                            MCFGChartEntry(
                                element, # element
                                  (trigger.left_side, (item1.symbol.string_spans, item2.symbol.string_spans)) # backpointer
                            )
                        )
                        new_triggers.add(element)
                    except:
                        pass
        return new_triggers

    @property
    def parses(self) -> set[Tree]:
        try:
            return self._parses
        except AttributeError:
            self._parses = self._construct_parses()
            return self._parses

    def _construct_parses(self, entry: Union['MCFGChartEntry', None] = None) -> set[Tree]:
        """Construct the parses implied by the chart

        Parameters
        ----------
        entry
        """
        if not entry:
            entry = self[((0,len(self._string)),)]
            if not entry:
                return set({})
            else:
                entry = list(entry)[0]
        

        match entry.backpointers:
            case ((_, (((i1, i2),),)),) if i2 == i1 + 1:
                return {Tree(entry.symbol, [Tree(self._string[i1], [])])}

        left_children = []
        right_children = []

        string, indices = entry.backpointers[0]

        left, right = indices

        left_children += [child for child in self[left]]
        right_children += [child for child in self[right]]

        left_trees = set()
        for child in left_children:
            left_trees |= self._construct_parses(child)

        right_trees = set()
        for child in right_children:
            right_trees |= self._construct_parses(child)
        
        return {Tree(entry.symbol, [left_tree, right_tree])
                for left_tree in left_trees
                for right_tree in right_trees}

Mode = Literal["recognize", "parse"]

class ContextFreeGrammarParser(ABC):

    def __init__(self, grammar: ContextFreeGrammar):
        self._grammar = grammar

    def __call__(self, string, mode="recognize"):
        if mode == "recognize":
            return self._recognize(string)
        elif mode == "parse":
            return self._parse(string)
        else:
            msg = 'mode must be "parse" or "recognize"'
            raise ValueError(msg)

class MCFGParser: # ¤ (ContextFreeGrammarParser):

    def __init__(self, grammar: MCFGGrammar):
        self._grammar = grammar

    def __call__(self, string, mode="recognize"):
        if mode == "recognize":
            return self._recognize(string)
        elif mode == "parse":
            return self._parse(string)
        else:
            msg = 'mode must be "parse" or "recognize"'
            raise ValueError(msg)

    def _recognize(self, string):
        chart = self._fill_chart(string)
        return self._grammar.start_variable in [entry.symbol.variable for entry in chart[((0,len(string)),)]]

    def _parse(self, string):
        chart = self._fill_chart(string)
        return chart.parses


    def _fill_chart(self, string):
        agenda = Agenda(self._grammar)
        chart = MCFGChart(string)

        while not agenda.is_empty:

            trigger = chart.add_item(agenda.select_item())
            agenda.add_items([rule
                              for rule in self._grammar.rules()
                              for element in trigger
                              if element.variable in rule.right_side])

        return chart

        # new agenda items are returned by chart.add_item, but have to be transformed to relevant left_side consequences

