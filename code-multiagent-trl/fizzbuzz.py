def fizzbuzz(n):
    """
    Print FizzBuzz for a given number n.
    
    Args:
        n (int): The number to check for FizzBuzz conditions.
    
    Returns:
        str: The FizzBuzz representation of the number.
    """
    if n % 15 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    else:
        return str(n)

def main():
    """
    Main function to run FizzBuzz for numbers 1 to 100.
    """
    for i in range(1, 101):
        result = fizzbuzz(i)
        print(f"{i}: {result}")

if __name__ == "__main__":
    main()