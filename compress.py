"""
Assignment 2 starter code
CSC148, Winter 2020
Instructors: Bogdan Simion, Michael Liut, and Paul Vrbik

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2020 Bogdan Simion, Michael Liut, Paul Vrbik, Dan Zingaro
"""
from __future__ import annotations
import time
from typing import Dict, Tuple
from utils import *
from huffman import HuffmanTree


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> Dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    d = {}
    for i in text:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d


def build_huffman_tree(freq_dict: Dict[int, int]) -> HuffmanTree:  # FIX THIS
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    if len(freq_dict) == 1:
        a = list(freq_dict.keys())
        b = (a[0] + 1) % 256
        return HuffmanTree(None, HuffmanTree(a[0]), HuffmanTree(b))
    lst = [(freq_dict[j], j) for j in freq_dict]
    lst.sort()
    lst2 = []
    for i in lst:
        lst2.append((i[0], HuffmanTree(i[1])))
    while len(lst2) > 1:
        za = lst2.pop(0)
        warudo = lst2.pop(0)
        a = HuffmanTree(None, za[1], warudo[1])
        lst2.append((za[0] + warudo[0], a))
        lst2.sort()
    real = lst2[0][1]
    __huffman_helper(real)
    return real


def __huffman_helper(real: HuffmanTree) -> None:
    """Helper function for build_huffman trees where it basically goes through
    the tree turning the internal nodes symbols to None"""
    if not real.is_leaf and real is not None:
        real.symbol = None
        __huffman_helper(real.left)
        __huffman_helper(real.right)


def get_codes(tree: HuffmanTree) -> Dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    d = {}
    __get_codes_helper2("", tree, d)
    if 'do i really need something here' in d:
        d.pop('do i really need something here')
    return d


def __get_codes_helper2(beg: str, tree: HuffmanTree, d: Dict) -> None:
    """My second attempt for get_codes helper, basic recursive function that
    gets me a dict. Mutates the dict in the helper so should save time """
    if tree is None:
        d['do i really need something here'] = 'i guess you do'
    elif tree.symbol is not None:
        d[tree.symbol] = beg
    else:
        __get_codes_helper2(beg + "0", tree.left, d)
        __get_codes_helper2(beg + "1", tree.right, d)


# this helper did not work but i kept it for  my reference. Took me hours to
# finally make get_codes_helper2, which in hindsight ticks me off because its
# just a basic recursive function. Oh well I learnt quite a bit from this
# function to say the least. Some mistakes included adding the '0' or '1' string
# from the wrong side. I tried to make
# a function that made use of a dict but turns out all I needed was a
# straightforward recursive function and a dict. This get_codes function really
# helped with my understanding of recursion.

# def __get_codes_helper(tree: HuffmanTree, d: Dict, beg: str):
#     if tree.left is not None:
#         if tree.left.symbol is not None:
#             # if tree.left.symbol not in d:
#             d[tree.left.symbol] = beg
#             # d[tree.left.symbol] += get_codes_helper(tree.left.symbol, d)
#             # else:
#             #     d[tree.left.symbol] = "0" + d[tree.left.symbol]
#         __get_codes_helper(tree.left, d, beg + "0")
#     if tree.right is not None:
#         if tree.right.symbol is not None:
#             # if tree.right.symbol not in d:
#             d[tree.right.symbol] = beg  # + d[tree.right.symbol]
#             # else:
#             #     d[tree.right.symbol] = "1" + d[tree.right.symbol]
#         __get_codes_helper(tree.right, d, beg + "1")
#
#     # if tree is not None: # do something
#     #     d[]


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    d = []
    for i in range(10000):
        d.append(i)
    __number_nodes_helper(tree, d)


def __number_nodes_helper(tree: HuffmanTree, ranger: list) -> None:
    """Helper to number_nodes"""
    if tree is not None and tree.left and tree.right:
        __number_nodes_helper(tree.left, ranger)
        __number_nodes_helper(tree.right, ranger)
        tree.number = ranger.pop(0)


def avg_length(tree: HuffmanTree, freq_dict: Dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.
    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    di = get_codes(tree)
    sum_freq = [freq_dict[i] for i in freq_dict.keys()]
    w_fsum = 0
    for j in sum_freq:
        w_fsum += j
    w_sum = 0
    for key, value in di.items():
        w_sum += freq_dict[key] * len(value)
    return w_sum / w_fsum


def compress_bytes(text: bytes, codes: Dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    za = ""
    for i in text:
        za += codes[i]
    lst = [bits_to_byte(za[j:j + 8]) for j in range(0, len(za), 8)]
    return bytes(lst)
    # My initial code below, which took 20 mins to compress Homer-Iliad.txt
    # List comprehension was very useful to cut down on the time to compress to
    # say the least.
    # za = ""
    # for i in range(len(text)):
    #     za += codes[text[i]]
    #
    # if len(za) > 8:
    #     jam = (len(za) // 8) + 1
    #     lst = []
    #     for i in range(jam):
    #         lst.append(bits_to_byte(za[0:8]))
    #         za = za[8:]
    #     return bytes(lst)
    # else:
    #     return bytes([bits_to_byte(za)])


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    # >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    # HuffmanTree(2, None, None))
    # >>> number_nodes(tree)
    # >>> list(tree_to_bytes(tree))
    # [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    lst = []
    __tree_to_bytes_helper(tree, lst)
    return bytes(lst)


def __tree_to_bytes_helper(tree: HuffmanTree, lster: list) -> None:
    """ Basically the tree to bytes function but it kept messing up due to
    me keep on reassigning lst as empty. So I put it in a helper and voila
    it works now."""
    if tree is not None and tree.left and tree.right:
        __tree_to_bytes_helper(tree.left, lster)
        __tree_to_bytes_helper(tree.right, lster)
        if tree.left.left is None and tree.left.right is None:
            lster.append(0)
            lster.append(tree.left.symbol)
        else:
            lster.append(1)
            lster.append(tree.left.number)
        if tree.right.left is None and tree.right.right is None:
            lster.append(0)
            lster.append(tree.right.symbol)
        else:
            lster.append(1)
            lster.append(tree.right.number)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree) +
              int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: List[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    # 0 is leaf, 1 is not a leaf
    root = node_lst[root_index]
    tree = HuffmanTree(None, None, None)
    if root.l_type == 1:
        a = node_lst[root.l_data]
        tree.left = __generate_tree_gen_help(node_lst, a)
    else:
        tree.left = HuffmanTree(root.l_data, None, None)
    if root.r_type == 1:
        a = node_lst[root.r_data]
        tree.right = __generate_tree_gen_help(node_lst, a)
    else:
        tree.right = HuffmanTree(root.r_data, None, None)
    return tree


def __generate_tree_gen_help(node_lst: list, a: ReadNode) -> HuffmanTree:
    """Helper for generate tree general"""
    buff = HuffmanTree(None, None, None)
    if a.l_type == 0:
        buff.left = HuffmanTree(a.l_data, None, None)
    else:
        b = node_lst[a.l_data]
        buff.left = __generate_tree_gen_help(node_lst, b)
    if a.r_type == 0:
        buff.right = HuffmanTree(a.r_data, None, None)
    else:
        b = node_lst[a.r_data]
        buff.right = __generate_tree_gen_help(node_lst, b)
    return buff


def generate_tree_postorder(node_lst: List[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    # 0 is leaf, 1 is not a leaf
    root_index += 1
    tree = HuffmanTree(None, None, None)
    for i in range(len(node_lst) - 1, -1, -1):
        a = node_lst[i]
        if a.l_type == 0 and a.r_type == 0:
            left_nodes = node_lst[0:i]
            right_nodes = node_lst[i:-1]
            if len(left_nodes) == 1:
                tree.left = HuffmanTree(None,
                                        HuffmanTree(left_nodes[0].l_data, None,
                                                    None),
                                        HuffmanTree(left_nodes[0].r_data, None,
                                                    None))
            else:
                buffet = __generate_tree_postorder_helper(left_nodes)
                tree.left = buffet
            if len(right_nodes) == 1:
                tree.right = HuffmanTree(None,
                                         HuffmanTree(right_nodes[0].l_data,
                                                     None, None),
                                         HuffmanTree(right_nodes[0].r_data,
                                                     None, None))
            else:
                buffet = __generate_tree_postorder_helper(right_nodes)
                tree.right = buffet
            break
    return tree


def __generate_tree_postorder_helper(node_lst: list) -> HuffmanTree:
    """Helper for generate tree postorder."""
    tree = HuffmanTree(None, None, None)
    for i in range(len(node_lst) - 1, -1, -1):
        a = node_lst[i]
        if a.l_type == 0 and a.r_type == 0:
            left_nodes = node_lst[0:i]
            right_nodes = node_lst[i:-1]
            if len(left_nodes) == 1:
                tree.left = HuffmanTree(None,
                                        HuffmanTree(left_nodes[0].l_data, None,
                                                    None),
                                        HuffmanTree(left_nodes[0].r_data, None,
                                                    None))
            else:
                buffet = __generate_tree_postorder_helper(left_nodes)
                tree.left = buffet
            if len(right_nodes) == 1:
                tree.right = HuffmanTree(None,
                                         HuffmanTree(right_nodes[0].l_data,
                                                     None, None),
                                         HuffmanTree(right_nodes[0].r_data,
                                                     None, None))
            else:
                buffet = __generate_tree_postorder_helper(right_nodes)
                tree.right = buffet
            break
    return tree


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.
    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    d = get_codes(tree)
    act = []
    bean = ""
    maxi = len(min(d.values(), key=len))
    maxer = maxi
    for i in text:
        bean += byte_to_bits(i)
        while maxi <= len(bean) and len(act) != size:
            if bean[:maxi] in d.values():
                act.append(
                    int(list(d.keys())[list(d.values()).index(bean[:maxi])]))
                bean, maxi = bean[maxi:], maxer
            else:
                maxi += 1
    return bytes(act)

    # initially i started from the largest len then went smaller at every
    # failure, but changed to starting at the smallest instead as im pretty sure
    # it would be more efficient.
    # Then as I was struggling to find ways to make my code more efficient,
    # I realised that i could start iterating not from 1, but the minimum
    # length available in the dict. So this would save more time the larger
    # the file is! Parrot.bmp compressed in 20 mins (!) on my pc, but when I
    # scp'd and ssh'd onto the lab computers it took me just 37 seconds.

    # This is pretty much the same code but with while loops
    # i = 0
    # while len(act) != size:
    #     bean += byte_to_bits(text[i])
    #     i += 1
    #     while maxi <= len(bean) and len(act) != size:
    #         if bean[:maxi] in d.values():
    #             act.append(
    #                 int(list(d.keys())[list(d.values()).index(bean[:maxi])]))
    #             bean, maxi = bean[maxi:], maxer
    #         else:
    #             maxi += 1


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: Dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    d = {}
    for key in freq_dict:
        d[freq_dict[key]] = key
    while len(d) != 0:
        __preorder_pls(tree, d, freq_dict)


def __preorder_pls(tree: HuffmanTree, freq_dict: Dict[int, int],
                   ff: Dict[int, int]) -> None:
    if tree.is_leaf and tree.symbol is not None:
        tree.symbol = freq_dict[max(freq_dict.keys())]
        freq_dict.pop(max(freq_dict.keys()))
    else:
        __preorder_pls(tree.left, freq_dict, ff)
        __preorder_pls(tree.right, freq_dict, ff)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input("Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        # cProfile.run('compress_file(fname, fname + ".huf")')
        print("Compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        # cProfile.run('decompress_file(fname, fname + ".orig")')
        print("Decompressed {} in {} seconds."
              .format(fname, time.time() - start))
