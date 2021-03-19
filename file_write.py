import shelve
db = shelve.open("cache/test")
db["test_score"] = [1, 2, 3, 4]
db.close()

db = shelve.open("cache/test")
a = db["test_score"]
print(a)
db.close()
