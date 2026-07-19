def longest_substring_without_repeating(s: str) -> int:
    """
    Find the length of the longest substring without repeating characters.
    Uses sliding window approach with O(n) time complexity.
    
    Args:
        s: Input string
    Returns:
        Length of longest substring without repeating characters
    """
    if not s:
        return 0
    
    seen = {}
    max_length = 0
    left = 0
    
    for right in range(len(s)):
        char = s[right]
        
        # If character is already in current window, shrink from left
        if char in seen and seen[char] >= left:
            left = seen[char] + 1
        
        # Update the position of the character
        seen[char] = right
        
        # Update max length
        current_length = right - left + 1
        max_length = max(max_length, current_length)
    
    return max_length


def test_longest_substring():
    """Test the function with various inputs."""
    test_cases = [
        ("abcabcbb", 3),  # "abc" is the longest
        ("bbbbb", 1),  # Only one unique character
        "", 0,  # Empty string
        "abcdef", 6,  # All unique
        "dvdf", 3,  # "vdf"
        "abcabc", 3,  # "abc"
    ]
    
    for s, expected in test_cases:
        result = longest_substring_without_repeating(s)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: '{s}' -> Output: {result} (Expected: {expected})")


if __name__ == "__main__":
    print("Testing Longest Substring Without Repeating Characters...")
    print("=" * 50)
    test_longest_substring()
    print("=" * 50)
    
    # Example usage
    s = "abcabcbb"
    print(f"\nInput: '{s}'")
    print(f"Longest substring length: {longest_substring_without_repeating(s)}")
    print(f"Longest substring: {'abc'}")
