# This file is a slightly modified version of 
# https://github.com/ML4SCI/SYMBA/blob/main/data_preprocessing/sympy_prefix/source/SympyPrefix.py

from itertools import repeat
import sympy as sp
from sympy import *
import numpy as np

import sys
sys.setrecursionlimit(2000)

# operators and operators_nargs are from 
# https://github.com/facebookresearch/SymbolicMathematics

operators = {
    # Elementary functions
    sp.Add: 'add',
    sp.Mul: 'mul',
    sp.Pow: 'pow',
    sp.exp: 'exp',
    sp.log: 'ln',
    sp.Abs: 'abs',
    sp.sign: 'sign',
    # Trigonometric Functions
    sp.sin: 'sin',
    sp.cos: 'cos',
    sp.tan: 'tan',
    sp.cot: 'cot',
    sp.sec: 'sec',
    sp.csc: 'csc',
    # Trigonometric Inverses
    sp.asin: 'asin',
    sp.acos: 'acos',
    sp.atan: 'atan',
    sp.acot: 'acot',
    sp.asec: 'asec',
    sp.acsc: 'acsc',
    # Hyperbolic Functions
    sp.sinh: 'sinh',
    sp.cosh: 'cosh',
    sp.tanh: 'tanh',
    sp.coth: 'coth',
    sp.sech: 'sech',
    sp.csch: 'csch',
    # Hyperbolic Inverses
    sp.asinh: 'asinh',
    sp.acosh: 'acosh',
    sp.atanh: 'atanh',
    sp.acoth: 'acoth',
    sp.asech: 'asech',
    sp.acsch: 'acsch',
    # Derivative
    sp.Derivative: 'derivative',
}

operators_inv = {operators[key]: key for key in operators}
operators_inv["mul("] = sp.Mul
operators_inv["add("] = sp.Add

operators_nargs = {
    # Elementary functions
    'mul(': -1,
    'add(': -1,
    'add': 2,
    'sub': 2,
    'mul': 2,
    'div': 2,
    'pow': 2,
    'rac': 2,
    'inv': 1,
    'pow2': 1,
    'pow3': 1,
    'pow4': 1,
    'pow5': 1,
    'sqrt': 1,
    'exp': 1,
    'ln': 1,
    'abs': 1,
    'sign': 1,
    # Trigonometric Functions
    'sin': 1,
    'cos': 1,
    'tan': 1,
    'cot': 1,
    'sec': 1,
    'csc': 1,
    # Trigonometric Inverses
    'asin': 1,
    'acos': 1,
    'atan': 1,
    'acot': 1,
    'asec': 1,
    'acsc': 1,
    # Hyperbolic Functions
    'sinh': 1,
    'cosh': 1,
    'tanh': 1,
    'coth': 1,
    'sech': 1,
    'csch': 1,
    # Hyperbolic Inverses
    'asinh': 1,
    'acosh': 1,
    'atanh': 1,
    'acoth': 1,
    'asech': 1,
    'acsch': 1,
    # Derivative
    'derivative': 2,
    # custom functions
    'f': 1,
    'g': 2,
    'h': 3,
}

masses_strings = [
        "m_e",
        "m_u",
        "m_d",
        "m_s",
        "m_c",
        "m_b",
        "m_t",
        ]

masses = [sp.Symbol(x) for x in masses_strings]

# these will be converted to the numbers format in `format_number`
integers_types = [
        sp.core.numbers.Integer,
        sp.core.numbers.One,
        sp.core.numbers.NegativeOne,
        sp.core.numbers.Zero,
        ]

numbers_types = integers_types + [sp.core.numbers.Rational,
        sp.core.numbers.Half, sp.core.numbers.Exp1, sp.core.numbers.Pi, "<class 'sympy.core.numbers.Pi'>",
        sp.core.numbers.ImaginaryUnit]

# don't continue evaluating at these, but stop
atoms = [
        str,
        sp.core.symbol.Symbol,
        sp.core.numbers.Exp1,
        sp.core.numbers.Pi,
        "<class 'sympy.core.numbers.Pi'>",
        ] + numbers_types

# variables or constants
variables = [
        'x',
        'y',
        'z',
        'a',
        'b',
        'c',
        'd',
        'E',
        'reg_prop',
        'm_s',
        'm_u'
        's_0',
        's_1',
        's_2',
        's_3',
        's_4',
        's_5',
        's_6',
        's_7',
        's_8',
        's_9',
        's_10',
        's_11',
        's_12',
        's_13',
        's_14',
        's_15',
        's_16',
        's_17',
        's_18',
        's_19',
        's_20',
        's_21',
        's_22',
        's_23',
        's_24',
        's_25',
        's_26',
        's_27',
        's_28',
        's_29',
        's_30',
        's_31',
        's_32',
        's_33',
        's_34',
        's_35',
        's_36',
        's_37',
        's_38',
        's_39',
        's_40',
        's_41',
        's_42',
        's_43',
        's_44',
        's_45',
        ]


def flatten(l, ltypes=(list, tuple)):
    """
    flatten a python list
    from http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
    """
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)


def pad_right(list, total_length=4, const=0):
    length = len(list)
    values_needed = total_length - length 
    if values_needed > 0:
        return np.pad(list, (0, values_needed), mode="constant", constant_values=const) 
    else:
        return list[0:total_length]

def sympy_tokenize(expr, tokens_list=[], depth=0):
    if (expr.func == sp.core.symbol.Symbol) | (expr.func == sp.core.numbers.Integer):
        to_append = expr
    else:
        to_append = expr.func
    tokens_list.append(to_append)
    for arg in expr.args:
        sympy_tokenize(arg, tokens_list, depth+1)
    return tokens_list

def sympy_tokenize_str(sentence):
    Xi_tokenized = sympy_tokenize(sentence, tokens_list=[])
    Xi_tokenized_str = [srepr(el) for el in Xi_tokenized]
    return Xi_tokenized_str

def key_to_index_lookup_safe(key_to_index, word, shift=0):
    """look up word in key_to_index, replace unknown with max_index+1"""
    number_of_keys = len(key_to_index)
    try:
        index = key_to_index[word]
    except:
        index = number_of_keys
    return index + shift

def vectorize_sentence(Xi, model):
    # 0 reserved for [end], so add 1 to index
    key_to_index = model.wv.key_to_index
    Xi_vectorized = [key_to_index_lookup_safe(key_to_index, word, shift=1) for word in Xi]
    return Xi_vectorized

def vectorize_ds(X_tokenized_str, model, sequence_length=25):
    X_vectorized = [ vectorize_sentence(sentence, model) for sentence in X_tokenized_str]
    # sequence_length = np.max([len(Xi) for Xi in X_vectorized]) + 5
    X_vectorized = [pad_right(Xi, sequence_length, const=0) for Xi in X_vectorized]
    return X_vectorized

def sympy_to_prefix(expression):
    """
    Recursively go from a sympy expression to a prefix notation.
    Returns a flat list of tokens.
    """
    return flatten(sympy_to_prefix_rec(expression, []))

def sympy_to_prefix_rec(expression, ret):
    """
    Recursively go from a sympy expression to a prefix notation.
    The operators all get converted to their names in the array `operators`.
    Returns a nested list, where the nesting basically stands for parentheses.
    Since in prefix notation with a fixed number of arguments for each function (given in `operators_nargs`),
    parentheses are not needed, we can flatten the list later.
    """
    if expression in [sp.core.numbers.Pi, sp.core.numbers.ImaginaryUnit]:
        f = expression
    else:
        f = expression.func
    if f in atoms:
        if type(expression) in numbers_types:
            return ret + format_number(expression)
        return ret+[str(expression)]
    f_str = operators[f]
    f_nargs = operators_nargs[f_str]
    args = expression.args
    if len(args) == 1 & f_nargs == 1: 
        ret = ret + [f_str]
        return sympy_to_prefix_rec(args[0], ret)
    if len(args) == 2:
        ret = ret + [f_str, sympy_to_prefix_rec(args[0], []), sympy_to_prefix_rec(args[1], [])]
    if len(args) > 2:
        args = list(map(lambda x: sympy_to_prefix_rec(x, []), args))
        ret = ret + repeat_operator_until_correct_binary(f_str, args)
    return ret 


def repeat_operator_until_correct_binary(op, args, ret=[]):
    """
    sympy is not strict enough with the number of arguments.
    E.g. multiply takes a variable number of arguments, but for 
    prefix notation it needs to ALWAYS have exactly 2 arguments

    This function is only for binary operators.

    Here I choose the convention as follows:
        1 + 2 + 3 --> + 1 + 2 3 
    
    This is the same convention as in https://arxiv.org/pdf/1912.01412.pdf
    on page 15.

    input:
        op: in string form as in the list `operators`
        args: [arg1, arg2, ...] arguments of the operator, e.c. [1, 2, x**2,
                ...]. They can have other things to be evaluated in them
        ret: the list you already have. Usually []. Watch out, I think one has to explicitely give [],
            otherwise somehow the default value gets mutated, which I find a strange python behavior.
    """

    is_binary = operators_nargs[op] == 2
    assert is_binary, "repeat_operator_until_correct_binary only takes binary operators" 

    if len(args) == 0:
        return ret
    elif len(ret) == 0:
        ret = [op] + args[-2:]
        args = args[:-2]
    else:
        ret = [op] + args[-1:] + ret
        args = args[:-1]

    return repeat_operator_until_correct_binary(op, args, ret)

def format_number(number):
    if type(number) in integers_types:
        return format_integer(number)
    elif type(number) == sp.core.numbers.Rational:
        return format_rational(number)
    elif type(number) == sp.core.numbers.Half:
        return format_half()
    elif type(number) == sp.core.numbers.Exp1:
        return format_exp1()
    elif type(number) == sp.core.numbers.Pi:
        return format_pi()
    elif type(number) == sp.core.numbers.ImaginaryUnit:
        return format_imaginary_unit()
    else:
        raise NotImplementedError

def format_exp1():
    return ['E']

def format_pi():
    return ['pi']

def format_imaginary_unit():
    return ['I']
        
def format_half():
    """
    for some reason in sympy 1/2 is its own object and not a rational.
    This function formats it correctly like `format_rational`
    """
    return ['mul'] + ['s+', '1'] + ['pow'] + ['s+', '2'] + ["s-", "1"]

def format_rational(number):
    # for some reason number.p is a string
    p = sp.sympify(number.p)
    q = sp.sympify(number.q)
    return ['mul'] + format_integer(p) + ['pow'] + format_integer(q) + ['s-', '1']

def format_integer(integer):
    """take a sympy integer and format it as in 
    https://arxiv.org/pdf/1912.01412.pdf

    input:
        integer: a `sympy.Integer` object, e.g. `sympy.Integer(-1)`

    output:
        [sign_token, digit0, digit1, ...]
        where sign_token is 's+' or 's-' 

    Example: 
        format_integer(sympy.Integer(-123))
        >> ['s-', '1', '2', '3']

    Implementation notes:
    Somehow Integer inherits from Rational in Sympy and a rational is p/q,
    so integer.p is used to extract the number.
    """
    # plus_sign = "s+"
    plus_sign = "s+"
    minus_sign = "s-"
    abs_num = abs(integer.p)
    is_neg = integer.could_extract_minus_sign()
    digits = list(str(abs_num))
    # digits = [str(abs_num)]

    if is_neg:
        ret = [minus_sign] + digits
    else:
        ret = [plus_sign] + digits

    return ret 


def prefix_to_sympy(expr_arr):
    if len(expr_arr) == 1:
        return parse_if_str(expr_arr[0])
    op_pos = rightmost_operand_pos(expr_arr)
    if (op_pos == -1) | (op_pos == len(expr_arr)):
        print("something went wrong, operator should not be at end of array")
    op = expr_arr[op_pos]
    if op in operators_inv.keys():
        num_args = operators_nargs[op]
        op = operators_inv[op]
        args = expr_arr[op_pos+1:op_pos+num_args+1]
        args = [parse_if_str(a) for a in args]
        func = op(*args)
        expr = expr_arr[0:op_pos] + [func] + expr_arr[op_pos+num_args+1:]
        return prefix_to_sympy(expr)

    elif (op == 's+') | (op == "s-"):
        # int_end_pos = rightmost_int_pos(expr_arr)
        string_end_pos = rightmost_string_pos(expr_arr)
        integer = unformat_integer(expr_arr[op_pos:string_end_pos+1])
        expr_arr_new = expr_arr[0:op_pos] + [integer] + expr_arr[string_end_pos+1:]
        return prefix_to_sympy(expr_arr_new)
    elif op in variables:
        op = sp.sympify(op)
        expr_arr_new = expr_arr[0:op_pos] + [op] + expr_arr[op_pos+1:]
        return prefix_to_sympy(expr_arr_new)

    return op

def parse_if_str(x):
    if isinstance(x, str):
        return sp.parsing.parse_expr(x)
    return x

# def rightmost_int_pos(expr_arr, pos=-1):
#     if isinstance(expr_arr[pos], str):
#         return len(expr_arr)+pos
#     else:
#         return rightmost_string_pos(expr_arr, pos-1)

def rightmost_string_pos(expr_arr, pos=-1):
    if isinstance(expr_arr[pos], str):
        return len(expr_arr)+pos
    else:
        return rightmost_string_pos(expr_arr, pos-1)


def rightmost_operand_pos(expr, pos=-1):
    operators = list(operators_inv.keys()) + ["s+", "s-"] + variables
    if expr[pos] in operators:
        return len(expr) + pos
    else:
        return rightmost_operand_pos(expr, pos-1)

def unformat_integer(arr):
    """
    inverse of the function format_integer.

    input:
        arr: array of strings just as the output of format_integer. E.g. ["s+", "4", "2"]

    output:
        the correspinding sympy integer, e.g. sympy.Integer(42) in the above example.

    The sign tokens are "s+" for positive integers and "s-" for negative. 0 comes with "s+", but does not matter.

    """
    sign_token = arr[0]
    ret = "-" if sign_token == "s-" else ""
    for s in arr[1:]:
        ret += str(s)

    return sp.parsing.parse_expr(ret)