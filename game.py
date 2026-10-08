import random

def number_guessing_game():
    """A simple number guessing game."""
    print("Welcome to the Number Guessing Game!")
    print("I'm thinking of a number between 1 and 100.")
    print("Try to guess it!")

    # Generate a random number between 1 and 100
    secret_number = random.randint(1, 100)
    attempts = 0

    while True:
        try:
            # Get user's guess
            guess = int(input("\nEnter your guess: "))
            attempts += 1

            # Check if guess is correct
            if guess < secret_number:
                print("Too low! Try again.")
            elif guess > secret_number:
                print("Too high! Try again.")
            else:
                print(f"\nCongratulations! You guessed the number in {attempts} attempts!")
                print(f"The number was {secret_number}.")
                break

        except ValueError:
            print("Please enter a valid number!")

if __name__ == "__main__":
    number_guessing_game()