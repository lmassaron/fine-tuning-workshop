import unittest
import sys
import os

# Resolve imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from longest_substring import longest_substring_without_repeating


class TestLongestSubstring(unittest.TestCase):
    """Unit tests for longest_substring_without_repeating function."""

    def test_empty_string(self):
        """Test with empty string."""
        self.assertEqual(longest_substring_without_repeating(""), 0)

    def test_single_character(self):
        """Test with single character."""
        self.assertEqual(longest_substring_without_repeating("a"), 1)

    def test_all_same_characters(self):
        """Test with all same characters."""
        self.assertEqual(longest_substring_without_repeating("aaaaa"), 1)

    def test_all_unique_characters(self):
        """Test with all unique characters."""
        self.assertEqual(longest_substring_without_repeating("abcdef"), 6)

    def test_repeating_characters(self):
        """Test with repeating characters."""
        self.assertEqual(longest_substring_without_repeating("abcabcbb"), 3)

    def test_repeating_characters_middle(self):
        """Test with repeating characters in middle."""
        self.assertEqual(longest_substring_without_repeating("dvdf"), 3)

    def test_repeating_characters_start(self):
        """Test with repeating characters at start."""
        self.assertEqual(longest_substring_without_repeating("abba"), 2)

    def test_repeating_characters_end(self):
        """Test with repeating characters at end."""
        self.assertEqual(longest_substring_without_repeating("abcba"), 3)

    def test_unicode_characters(self):
        """Test with unicode characters."""
        self.assertEqual(longest_substring_without_repeating("你好世界"), 4)

    def test_mixed_case(self):
        """Test with mixed case characters."""
        self.assertEqual(longest_substring_without_repeating("AbCdEf"), 6)

    def test_long_string(self):
        """Test with longer string."""
        self.assertEqual(longest_substring_without_repeating("abcdefg"), 7)

    def test_long_string_with_repeats(self):
        """Test with longer string with repeats."""
        self.assertEqual(longest_substring_without_repeating("abcdefgabcdefg"), 7)


if __name__ == "__main__":
    unittest.main()
