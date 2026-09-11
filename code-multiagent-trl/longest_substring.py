def longest_substring_without_repeating_chars(s):
    """
    Find the length of the longest substring without repeating characters.
    
    Args:
        s (str): The input string to analyze.
    
    Returns:
        int: The length of the longest substring without repeating characters.
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
        result = longest_substring_without_repeating_chars(test)
        print(f"Input: '{test}' -> Length: {result}")
