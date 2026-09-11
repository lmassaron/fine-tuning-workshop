import unittest
from longest_substring import longest_substring_without_repeating_chars

class TestLongestSubstring(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(longest_substring_without_repeating_chars(''), '')
    
    def test_single_character(self):
        self.assertEqual(longest_substring_without_repeating_chars('a'), 'a')
    
    def test_all_unique_chars(self):
        self.assertEqual(longest_substring_without_repeating_chars('abcdef'), 'abcdef')
    
    def test_with_repeating_chars(self):
        self.assertEqual(longest_substring_without_repeating_chars('abcabc'), 'abc')
    
    def test_all_same_chars(self):
        self.assertEqual(longest_substring_without_repeating_chars('aaaa'), 'a')
    
    def test_with_spaces(self):
        self.assertEqual(longest_substring_without_repeating_chars('a b c'), 'a b c')
    
    def test_with_punctuation(self):
        self.assertEqual(longest_substring_without_repeating_chars('a.b.c'), 'a.b.c')

if __name__ == '__main__':
    unittest.main()