from functools import cached_property
class Test(object):
    def __init__(self):
        print("Initializing instance of Test class.")

    @cached_property
    def value(self):
        return 5

    @cached_property
    def other_value(self):
        return 10

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


if __name__ == "__main__":
    from collections import Counter

    test1 = Test()
    test2 = Test()
    test2.value = 6
    myCounter= Counter([test1, test2])
    print(myCounter)
    test2.value = 5
    print(myCounter)
    new_counter= Counter({6: 3, 4: 2, 5: 1})
    new_counter[7]=5
    print(new_counter)

    # from operator import attrgetter
    # test = Test()
    # print(test.value)
    # test.value = 6
    # print(test.value)
    # print(attrgetter('value', 'other_value')(test))
    # test.final_value = 7
    # print(test.final_value)
    import numpy as np
    import numpy.lib.recfunctions as nlr

    a = np.random.randint(0, high=10, size=(10, 5, 2), dtype=np.uint16)
    def last_dim_to_tuple(a):
        # return np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.descr * a.shape[-1])))[..., 0]
        descr= [(np.void, '<u2')]
        descr[]
        return np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.descr * a.shape[-1])))[..., 0]
    b = last_dim_to_tuple(a)
    # c = nlr.unstructured_to_structured(b)
    # d = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.descr * a.shape[-1])))[..., 0]
    print(b)
    # print(b)
    # print(nlr.unstructured_to_structured(a))

