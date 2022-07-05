import difflib
import xml.etree.ElementTree as ET
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
import re
import parsing_utils.constants as const
import word2number.w2n as w2n
from typing import Optional, Dict, List
import logging
from bs4 import BeautifulSoup
import lxml
import pdb

# dataclass that stores each term in a declaration
@dataclass
class Term:
    # original name
    name: str
    # index in order mapping
    index: int
    # the constant that multiplies the term
    value: float = None


@dataclass
class Declaration:
    # minimize/maximize or <=/>=
    direction: str
    # mapping of term name to each term involved in the declaration
    terms: Dict[str, Term]
    # order mapping
    entities: Dict[str, int]


@dataclass
class ObjectiveDeclaration(Declaration):
    # name of variable being minimized/maximized
    name: str


@dataclass
class ConstraintDeclaration(Declaration):
    # type of constraint, e.g., linear
    type: str
    # the constant on the right side of the constraint
    limit: float
    # >= or <= for constraint
    operator: str
    # using ordered dict here as order determines order of operations in balance control constraints
    terms: Dict[str, Term]


@dataclass
class ProblemFormulation:
    objective: ObjectiveDeclaration
    constraints: List[ConstraintDeclaration]
    # order mapping mapping each entity to its index
    entities: Dict[str, int]


@dataclass
class CanonicalFormulation:
    objective: np.ndarray
    constraints: np.ndarray


def convert_to_canonical(formulation: ProblemFormulation) -> CanonicalFormulation:
    n_entities = max(formulation.entities.values()) + 1 if len(formulation.entities) else 0
    constraints = []
    objective = np.zeros(n_entities)
    for k, v in formulation.objective.terms.items():
        # check if value was given in formulation
        objective[v.index] = v.value if v.value is not None else np.nan

    for constraint in formulation.constraints:

        row = np.ones(n_entities + 1)
        # compute everything as <= at first
        if constraint.type == const.SUM_CONSTRAINT:
            # x + y <= 150
            row[-1] = constraint.limit

        elif constraint.type == const.LOWER_BOUND or constraint.type == const.UPPER_BOUND:
            # x <= 50
            # compute as upper bound and flip later for lower bound if necessary
            row *= 0
            for k, v in constraint.terms.items():
                row[v.index] = 1
            row[-1] = constraint.limit
        elif constraint.type == const.LINEAR_CONSTRAINT:
            # 2x + 3y <= 20
            for k, v in constraint.terms.items():
                # check if value was given in formulation
                row[v.index] = v.value if v.value is not None else np.nan
            row[-1] = constraint.limit
        elif constraint.type == const.RATIO_CONTROL_CONSTRAINT:
            # x <= 0.7 (x + y)
            row *= -constraint.limit
            for k, v in constraint.terms.items():
                row[v.index] = 1 - constraint.limit
            row[-1] = 0
        elif constraint.type == const.BALANCE_CONSTRAINT_1:
            # x <= 3y
            row *= 0
            # y should be first term, but we will parse whichever term that has value as y
            for i, (k, v) in enumerate(constraint.terms.items()):
                if v.value is not None:
                    row[v.index] = - v.value
                else:
                    row[v.index] = 1
        elif constraint.type == const.BALANCE_CONSTRAINT_2:
            # x <= y
            for i, (k, v) in enumerate(constraint.terms.items()):
                # y is first term
                if i == 0:
                    row[v.index] = - 1
                elif i == 1:
                    row[v.index] = 1
        # flip if >=
        if constraint.operator == const.GT:
            row *= -1
        constraints.append(row)

    return CanonicalFormulation(objective, np.asarray(constraints))


class Parser:
    def __init__(self, print_errors=True) -> None:
        self.print_errors = print_errors

    def parse_number(self, x: str) -> float:
        """

        :param x: any number-like string
        :return: best-effort attempt at converting the string to a number; returns 0 if it could not parse
        """
        x = x.strip().replace(',', '').strip(';,$')
        # remove extra periods from the end
        x = re.sub('\.+$', '', x)
        multiplier = 1
        x_out = 0

        # convert percent, cents to fractional
        x_sub = re.sub('(percent)|%|¢', '', x)
        if x_sub != x:
            multiplier = 1 / 100
            x = x_sub
        try:
            x_out = float(x) * multiplier
        except ValueError:
            # see if there are any digits and strip everything else out
            if re.search('[\d.]+', x):
                res = re.sub('[^0-9.]', '', x)
                x_out = float(res) * multiplier
                if self.print_errors:
                    logging.info(
                        f'Non-numeric input \"{x}\" converted to \"{x_out}\" by filtering out non-number characters')
            else:
                try:
                    # see if it is in predefined constants
                    if x in const.NUMS_DICT:
                        x_out = const.NUMS_DICT[x] * multiplier
                    else:
                        x_out = float(w2n.word_to_num(x)) * multiplier
                        if self.print_errors:
                            logging.info(f'Non-numeric input \"{x}\" converted to {x_out} with w2n')
                except ValueError:
                    if self.print_errors:
                        logging.warning(f'Could not convert word \"{x}\" to number')
        return x_out

    # for general text parsing
    def parse_text(self, x: str) -> str:
        return x.strip()

    # for fuzzy searching the order mapping
    # will only strip if order mapping is empty
    def parse_entity(self, x: str, order_mapping: dict) -> str:
        x = x.strip()
        if x in order_mapping:
            return x
        elif len(order_mapping):
            # try to find closest match in the order mapping
            best_similarity = 0
            best_match = x
            for k in order_mapping:
                # use lower case as this metric penalizes for different cases
                score = self.similarity(x, k)
                if score > best_similarity:
                    best_similarity = score
                    best_match = k
            return best_match
        else:
            return x

    # string similarity metric to perform fuzzy search
    def similarity(self, a: str, b: str):
        a = a.strip().lower()
        b = b.strip().lower()
        # ignore spaces with isjunk
        sm = difflib.SequenceMatcher(isjunk=lambda x: x in " \t", a=a, b=b)
        return sm.ratio()

    # to be overridden if a custom parser is required
    def parse(self, data: object, order_mapping: Optional[dict] = None) -> ProblemFormulation:
        """

        :param data: parsing_utils to parse, typically a string or dict
        :param order_mapping: mapping of variables to an index to convert into canonical form; should be given in dataset
        """
        pass


# Parses the XML-like intermediate outputs of a model
class ModelOutputXMLParser(Parser):
    def xmltree(self, data: str) -> Optional[ET.Element]:
        # fix mismatched tags
        bs = BeautifulSoup(f'<s>{data}</s>', 'xml')
        # remove empty elements
        for x in bs.find_all():
            if len(x.get_text(strip=True)) == 0:
                x.extract()
        return ET.fromstring(str(bs))

    def parse(self, data: str, order_mapping=None) -> ProblemFormulation:
        try:
            root = self.xmltree(data)
            # use iter instead of find_all in case of weird nesting
            declarations = root.iter('DECLARATION')
            # assuming one objective
            objective = None
            constraints = []
            # find objective first
            for declaration in declarations:
                if declaration.find('OBJ_DIR') is not None:
                    objective = self.parse_objective(declaration, order_mapping)
                    break
            # then do the constraints
            for declaration in declarations:
                if declaration.find('CONST_DIR') is not None:
                    try:
                        constraints.append(self.parse_constraint(declaration, objective.entities))
                    except ValueError as e:
                        if self.print_errors:
                            logging.warning(f'Could not parse constraint, skipping: {e}')
            return ProblemFormulation(objective, constraints, entities=objective.entities)
        except Exception as e:
            if self.print_errors:
                logging.warning(
                    f'Could not parse text \"{data}\".\nPlease check that your model output is parsable XML.\nProceeding with empty ProblemFormulation.')
            # cannot be parsed, returning an empty ProblemFormulation
            return ProblemFormulation(ObjectiveDeclaration('', {}, {}, ''), [], entities={})

    def parse_objective(self, root: ET.Element, order_mapping) -> ObjectiveDeclaration:
        obj_dir = ''
        obj_name = ''
        variables = {}
        entities = order_mapping if order_mapping is not None else {}
        current_var = None
        count = 0
        for node in root:
            if node.tag == 'OBJ_DIR':
                obj_dir = self.parse_text(node.text)
            elif node.tag == 'OBJ_NAME':
                obj_name = self.parse_entity(node.text, entities)
            elif node.tag == 'VAR':
                if current_var is not None:
                    # case where VAR does not have a PARAM
                    variables[current_var.name] = current_var
                if order_mapping is None:
                    # if no order mapping try to make one
                    name = self.parse_entity(node.text, {})
                    current_var = Term(name=name, index=count)
                    entities[name] = count
                    count += 1
                else:
                    # use order mapping if it exists
                    name = self.parse_entity(node.text, entities)
                    current_var = Term(name=name, index=entities[name])
            elif node.tag == 'PARAM':
                current_var.value = self.parse_number(node.text)
                variables[current_var.name] = current_var
                current_var = None

        return ObjectiveDeclaration(name=obj_name, direction=obj_dir, terms=variables, entities=entities)

    def parse_constraint(self, root: ET.Element, entities: dict) -> ConstraintDeclaration:
        const_dir = ''
        limit = ''
        const_type = ''
        operator = ''
        variables = OrderedDict()
        current_var = None
        for node in root:
            if node.tag == 'CONST_DIR':
                const_dir = self.parse_text(node.text)
            elif node.tag == 'OPERATOR':
                operator = self.parse_text(node.text)
            elif node.tag == 'LIMIT':
                limit = self.parse_number(node.text)
            elif node.tag == 'CONST_TYPE':
                const_type = self.parse_text(node.text)
            elif node.tag == 'VAR':
                if current_var:
                    variables[current_var.name] = current_var
                name = self.parse_entity(node.text, entities)
                current_var = Term(name=name, index=entities[name])
            elif node.tag == 'PARAM':
                current_var.value = self.parse_number(node.text)
                variables[current_var.name] = current_var
                current_var = None

        if current_var is not None and current_var.name not in variables:
            variables[current_var.name] = current_var
        if const_type == const.BALANCE_CONSTRAINT_1 or const_type == const.BALANCE_CONSTRAINT_2:
            if len(variables) != 2:
                raise ValueError(
                    f'Balance constraint has incorrect number of variables (got: {len(variables)}, expected: 2): {ET.tostring(root)}')
            if const_type == const.BALANCE_CONSTRAINT_1 and list(variables.values())[0].value is None:
                raise ValueError(
                    f'Balance constraint xby has missing value for y: {ET.tostring(root)}')
        return ConstraintDeclaration(direction=const_dir, limit=limit, operator=operator,
                                     type=const_type, terms=variables, entities=entities)

    def parse_file(self, fname: str, order_mapping = None) -> Optional[ProblemFormulation]:
        with open(fname, 'r') as fd:
            data = fd.read()
            return self.parse(data, order_mapping)


# Parses the JSON formatted training examples
class JSONFormulationParser(Parser):

    def parse(self, data: dict, order_mapping=None) -> ProblemFormulation:
        try:
            # get actual data, top level is a numeric key pointing to data
            key = ''
            for k, v in data.items():
                data = v
                key = k

            order_mapping = data['order_mapping'] if order_mapping is None else order_mapping
            objective = self.parse_objective(data['obj_declaration'], data['vars'], order_mapping)
            constraints = []
            for constraint in data['const_declarations']:
                constraints.append(self.parse_constraint(constraint, objective.entities, order_mapping))
            return ProblemFormulation(objective, constraints, order_mapping)
        except:
            logging.warning(
                f'Could not parse example {key}: {data}.\nProceeding with empty ProblemFormulation.')
            # cannot be parsed, returning an empty ProblemFormulation
            return ProblemFormulation(ObjectiveDeclaration('', {}, {}, ''), [], entities={})

    def parse_objective(self, data: dict, vars: dict, order_mapping: dict) -> ObjectiveDeclaration:
        terms = {}

        if 'terms' in data:
            for i, (k, v) in enumerate(data['terms'].items()):
                terms[k] = Term(name=k, index=order_mapping[k], value=self.parse_number(v))
        else:
            # assume values are 1 if there is no terms mapping
            for var in vars:
                terms[var] = Term(name=var, index=order_mapping[var], value=1)

        return ObjectiveDeclaration(name=data['name'],
                                    direction=data['direction'],
                                    terms=terms,
                                    entities=order_mapping)

    def parse_constraint(self, data: dict, entities: dict, var_dict: dict) -> ConstraintDeclaration:
        terms = OrderedDict()
        limit = self.parse_number(data['limit']) if 'limit' in data else 0
        constraint_type = const.TYPE_DICT[data['type']]
        direction = self.parse_text(data['direction'])
        operator = self.parse_text(data['operator'])

        if 'terms' in data:
            for k, v in data['terms'].items():
                terms[k] = Term(name=k, index=entities[k], value=self.parse_number(v))
        elif constraint_type == const.UPPER_BOUND or constraint_type == const.LOWER_BOUND or constraint_type == const.RATIO_CONTROL_CONSTRAINT:
            k = data['var']
            terms[k] = Term(name=k, index=entities[k])
        elif constraint_type == const.SUM_CONSTRAINT:
            # no parsing for terms needed here
            pass
        elif constraint_type == const.BALANCE_CONSTRAINT_1:
            k = data['y_var']
            terms[k] = Term(name=k, index=entities[k], value=self.parse_number(data['param']))
            k = data['x_var']
            terms[k] = Term(name=k, index=entities[k])
        elif constraint_type == const.BALANCE_CONSTRAINT_2:
            k = data['y_var']
            terms[k] = Term(name=k, index=entities[k])
            k = data['x_var']
            terms[k] = Term(name=k, index=entities[k])
        else:
            logging.info(f'Could not find terms in {data}')

        return ConstraintDeclaration(type=constraint_type,
                                     direction=direction,
                                     operator=operator,
                                     entities=entities,
                                     limit=limit,
                                     terms=terms)
