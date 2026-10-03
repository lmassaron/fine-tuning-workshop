def test_no_repeating_characters(): 
    assert longest_substring('abcde') == 5


def test_repeating_characters(): 
    assert longest_substring('abcabc') == 3


def test_empty_string(): 
    assert longest_substring('') == 0


def test_all_repeating_characters(): 
    assert longest_substring('aaaaa') == 1


def test_one_character(): 
    assert longest_substring('a') == 1