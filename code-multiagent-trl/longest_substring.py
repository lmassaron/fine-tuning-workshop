def longest_substring(s):
    """
    Find the length of the longest substring without repeating characters.
    
    Args:
        s: Input string
        
    Returns:
        int: Length of the longest substring without repeating characters
    """
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


# Compatibility alias
longest_substring_without_repeating = longest_substring


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "abcabcbb",
        "bbbbb",
        "pwwkew",
        "",
        "abcdef"
    ]
    
    for test in test_cases:
        print(f"Input: '{test}' -> Output: {longest_substring(test)}")
