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
    
    char_set = set()
    max_length = 0
    start = 0
    
    for end in range(len(s)):
        while s[end] in char_set:
            char_set.remove(s[start])
            start += 1
        char_set.add(s[end])
        max_length = max(max_length, end - start + 1)
    
    return max_length


def longest_substring_without_repeating_chars_v2(s: str) -> int:
    """
    Alternative implementation using sliding window with dictionary.
    
    Args:
        s: Input string
    
    Returns:
        int: Length of the longest substring without repeating characters
    """
    if not s:
        return 0
    
    char_map = {}
    max_length = 0
    start = 0
    
    for end in range(len(s)):
        if s[end] in char_map and char_map[s[end]] >= start:
            start = char_map[s[end]] + 1
        char_map[s[end]] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length


# Aliases for compatibility
longest_substring_without_repeating = longest_substring_without_repeating_chars
lengthOfLongestSubstring = longest_substring_without_repeating_chars
