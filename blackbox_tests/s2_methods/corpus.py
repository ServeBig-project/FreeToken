"""Frozen public inputs and executable task contracts, independent of serving code."""

PERFORMANCE = [
    ("sky", "Explain in detail why the sky is blue and how scattering varies with wavelength:"),
    ("merge_sort", "Write a complete, well-commented Python implementation of stable merge sort, including a merge helper and a worked example:"),
]

CALIBRATION = [
    "Explain how database indexes speed up reads and affect writes, with practical examples:",
    "Write a Python example that reads JSON records and summarizes numeric measurements, explaining the design:",
    "Explain the difference between a process and a thread and describe a practical producer-consumer pipeline:",
]

TASKS = [
    {"name": "merge_intervals", "spec": "Define merge_intervals(intervals). Input is a list of closed integer [start,end] intervals, start<=end. Return a sorted list merging overlapping or touching intervals.", "cases": [
        ([[]], []),
        ([[[1, 3], [2, 6], [8, 10], [10, 12]]], [[1, 6], [8, 12]]),
        ([[[5, 7], [1, 2], [3, 4]]], [[1, 2], [3, 4], [5, 7]]),
        ([[[1, 10], [2, 3], [4, 8]]], [[1, 10]]),
        ([[[2, 2], [2, 3]]], [[2, 3]]),
    ]},
    {"name": "subtract_intervals", "spec": "Define subtract_intervals(base, cuts). base is one half-open integer [start,end] interval; cuts is a list of half-open intervals, possibly unsorted, overlapping, or outside base. Return the sorted nonempty intervals remaining after removing their union. An empty base returns [].", "cases": [
        ([[0, 10], []], [[0, 10]]),
        ([[0, 10], [[2, 4], [6, 8]]], [[0, 2], [4, 6], [8, 10]]),
        ([[0, 10], [[8, 15], [-2, 3], [2, 6]]], [[6, 8]]),
        ([[0, 10], [[-1, 11]]], []),
        ([[3, 3], [[0, 5]]], []),
    ]},
    {"name": "topological_layers", "spec": "Define topological_layers(n, edges). Nodes are 0..n-1 and directed edges are [u,v], with duplicates ignored. Repeatedly collect ALL remaining zero-indegree nodes into a sorted layer, remove that entire layer, and continue. Return the list of layers, or None if any cycle exists. Include isolated nodes.", "cases": [
        ([0, []], []),
        ([4, [[0, 2], [1, 2], [2, 3]]], [[0, 1], [2], [3]]),
        ([4, []], [[0, 1, 2, 3]]),
        ([3, [[0, 1], [1, 0]]], None),
        ([3, [[0, 1], [0, 1]]], [[0, 2], [1]]),
    ]},
    {"name": "lru_trace", "spec": "Define lru_trace(capacity, requests). capacity>=1 and requests is a list of integer keys. Simulate an initially empty LRU cache: a hit becomes most recent; a miss inserts the key and evicts the least recent key if needed. Return [miss_count, final_keys_oldest_to_newest].", "cases": [
        ([2, [1, 2, 1, 3]], [3, [1, 3]]),
        ([1, [1, 1, 2, 1]], [3, [1]]),
        ([3, []], [0, []]),
        ([3, [1, 2, 3, 1, 4]], [4, [3, 1, 4]]),
        ([2, [1, 2, 3, 2, 4, 2]], [4, [4, 2]]),
    ]},
    {"name": "group_anagrams", "spec": "Define group_anagrams(words). Group case-sensitive strings that contain the same characters with the same multiplicities. Preserve input order inside each group and order groups by their first appearance. Return a list of lists.", "cases": [
        ([[]], []),
        ([["eat", "tea", "tan", "ate", "nat", "bat"]], [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]),
        ([["", ""]], [["", ""]]),
        ([["ab", "ba", "a", "ab"]], [["ab", "ba", "ab"], ["a"]]),
        ([["A", "a", "Aa", "aA"]], [["A"], ["a"], ["Aa", "aA"]]),
    ]},
    {"name": "sliding_max", "spec": "Define sliding_max(values, window). Return the maximum for each contiguous FULL window of a list of integers. window>=1. Return [] for empty input or when window is larger than the list.", "cases": [
        ([[], 1], []),
        ([[1, 3, -1, -3, 5, 3, 6, 7], 3], [3, 3, 5, 5, 6, 7]),
        ([[-5, -2, -2, -8], 2], [-2, -2, -2]),
        ([[4, 1, 9], 1], [4, 1, 9]),
        ([[2, 2, 1], 4], []),
    ]},
    {"name": "normalize_path", "spec": "Define normalize_path(path) for an absolute Unix path. Collapse repeated slashes, remove '.', resolve '..' by removing one previous component, and clamp at root. Return an absolute path with no trailing slash except root. This is string normalization, without filesystem access.", "cases": [
        (["/"], "/"),
        (["/a//b/./c/../"], "/a/b"),
        (["/../../a"], "/a"),
        (["/a/b/../../c"], "/c"),
        (["/a/../b/../../"], "/"),
    ]},
    {"name": "parse_csv_line", "spec": "Define parse_csv_line(line). Parse one valid CSV record with comma separators, double-quoted fields, and doubled quotes as escaped quotes. There are no embedded newlines. Preserve empty fields; an empty input represents one empty field and must return ['']. Return a list of strings. Python's standard library is allowed.", "cases": [
        (["a,b,c"], ["a", "b", "c"]),
        (["a,,c,"], ["a", "", "c", ""]),
        (['"a,b","he said ""hi"""'], ["a,b", 'he said "hi"']),
        ([""], [""]),
        (['"",x'], ["", "x"]),
    ]},
]


def coding_prompt(task):
    instruction = (task["spec"] + " Return only executable Python 3 code defining the function. "
                   "Use only the standard library. Do not use input(), print(), markdown, or explanations. /no_think")
    return "<|im_start|>user\n" + instruction + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
