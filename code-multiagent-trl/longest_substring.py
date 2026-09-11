def longest_substring_without_repeating_chars(s: str) -> int:
    """
    Find the length of the longest substring without repeating characters.
    
    Args:
        s: Input string
    
    Returns:
        int: Length of the longest substring without repeating characters
    """
    if not s:
        return 0
    
    char_index = {}
    max_length = 0
    start = 0
    
    for i, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = i
        max_length = max(max_length, i - start + 1)
    
    return max_length


def longest_substring_without_repeating_chars_v2(s: str) -> int:
    """
    Alternative implementation using sliding window with set.
    
    Args:
        s: Input string
    
    Returns:
        int: Length of the longest substring without repeating characters
    """
    if not s:
        return 0
    
    char_set = set()
    max_length = 0
    start = 0
    
    for i, char in enumerate(s):
        while char in char_set:
            char_set.remove(s[start])
            start += 1
        char_set.add(char)
        max_length = max(max_length, i - start + 1)
    
    return max_length
