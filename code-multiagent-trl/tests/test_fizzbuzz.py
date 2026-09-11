import unittest
from fizzbuzz import fizzbuzz

class TestFizzBuzz(unittest.TestCase):
    def test_fizzbuzz_1(self):
        self.assertEqual(fizzbuzz(1), 1)
    
    def test_fizzbuzz_3(self):
        self.assertEqual(fizzbuzz(3), 'Fizz')
    
    def test_fizzbuzz_5(self):
        self.assertEqual(fizzbuzz(5), 'Buzz')
    
    def test_fizzbuzz_15(self):
        self.assertEqual(fizzbuzz(15), 'FizzBuzz')
    
    def test_fizzbuzz_10(self):
        self.assertEqual(fizzbuzz(10), 'Buzz')
    
    def test_fizzbuzz_20(self):
        self.assertEqual(fizzbuzz(20), 'Fizz')
    
    def test_fizzbuzz_30(self):
        self.assertEqual(fizzbuzz(30), 'FizzBuzz')
    
    def test_fizzbuzz_100(self):
        self.assertEqual(fizzbuzz(100), 'FizzBuzz')