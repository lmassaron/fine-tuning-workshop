import unittest
import sys
import os

# Resolve imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from longest_substring import longest_substring_without_repeating

class TestLongestSubstring(unittest.TestCase):
    def test_example1(self):
        """Test with example from problem description"""
        s = "abcabcbb"
        self.assertEqual(longest_substring_without_repeating(s), 3)
    
    def test_example2(self):
        """Test with another example"""
        s = "bbcaacbb"
        self.assertEqual(longest_substring_without_repeating(s), 3)
    
    def test_example3(self):
        """Test with all unique characters"""
        s = "abcdef"
        self.assertEqual(longest_substring_without_repeating(s), 6)
    
    def test_example4(self):
        """Test with all same characters"""
        s = "aaaaa"
        self.assertEqual(longest_substring_without_repeating(s), 1)
    
    def test_example5(self):
        """Test with empty string"""
        s = ""
        self.assertEqual(longest_substring_without_repeating(s), 0)

if __name__ == "__main__":
    unittest.main()
