def is_prime(number):
    if number < 2:
        return False
    if number % 2 == 0:
        return number == 2
    limit = int(number ** 0.5)
    divisor = 3
    while divisor <= limit:
        if number % divisor == 0:
            return False
        divisor += 2
    return True

first_ten_primes = []
candidate = 2
while len(first_ten_primes) < 10:
    if is_prime(candidate):
        first_ten_primes.append(candidate)
    candidate += 1

print(first_ten_primes)
