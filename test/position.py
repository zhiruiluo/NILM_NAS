def position(a, /, b, *, c=1):
    return a + b + c

def test_1():
    position(1,b=2,c=3)
    position(1,b=2,c=3)