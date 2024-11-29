# INDEXING --> array name[index]

shopper_guy = [1, 2, 3, 4, 5, 6, 7, 8, 9]

""" shopper_girl = ['a', 'b', 'c', 'd']
# 0 1 2 3
rafay_test = ["Rafay", "Umer", "Ayat"] """

# TRAVERSAL --> GOING THROUGH THE ARRAY USING A FOR LOOP

SIZE = 9
index = 0
count = 0

for index in range(SIZE):
    if shopper_guy[index] > 5:
        count = count + 1

print("Count:", count)


"""
size = 9
count <- 0

for index <- 1 to size:
    if shopper_guy[index] > 5
        then
            count <- count + 1
next index

output "count:", count

"""



"""
sum <- 0

for index <- 1 to SIZE:
    sum <- sum + shopper_guy[index]
next index

output "SUM:", sum

"""

"""
for index <- 1 to SIZE:
    output shopper_guy[index]
next index

"""
