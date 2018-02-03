import unittest
import tkinter

from utils.util import SteppedIntVar

master = tkinter.Tk()


def check_list_io(i_: list, o_: list, tc: unittest.TestCase, t: SteppedIntVar):
    if len(i_) != len(o_):
        raise ValueError
    for index, _ in enumerate(i_):
        t.set(i_[index])
        tc.assertEqual(o_[index], t.get())


class MyTestCase(unittest.TestCase):
    def test_steppedIntVar(self):

        # Case1: int list
        i = [0, 1, 2, 3, 4, 5]
        o = [0, 1, 2, 3, 4, 5]
        siv = SteppedIntVar(starting=0, step=1)
        check_list_io(i, o, self, siv)

        # Case2:
        i = [1, 2, 3, 4, 5]
        o = [1, 2, 3, 4, 5]
        siv = SteppedIntVar(starting=0, step=1)
        check_list_io(i, o, self, siv)
        siv = SteppedIntVar(starting=1, step=1)
        check_list_io(i, o, self, siv)

        # Case3: odd list
        siv = SteppedIntVar(starting=1, step=2)
        i = [1, 3, 5, 7, 9, 11]
        o = [1, 3, 5, 7, 9, 11]
        check_list_io(i, o, self, siv)
        i = [2, 4, 6, 8, 10]
        o = [3, 5, 7, 9, 11]
        check_list_io(i, o, self, siv)
        i = [1, 2, 3, 4, 5]
        o = [1, 3, 3, 5, 5]
        check_list_io(i, o, self, siv)

        # Case4:
        i = [-0, -1, -2, -3, -4, -5]
        o = [-0, -1, -2, -3, -4, -5]
        siv = SteppedIntVar(starting=0, step=1)
        check_list_io(i, o, self, siv)

        # Case5: odd list
        siv = SteppedIntVar(starting=-11, step=2)
        i = [1, 3, 5, 7, 9, 11]
        o = [1, 3, 5, 7, 9, 11]
        check_list_io(i, o, self, siv)
        i = [2, 4, 6, 8, 10]
        o = [3, 5, 7, 9, 11]
        check_list_io(i, o, self, siv)
        i = [1, 2, 3, 4, 5]
        o = [1, 3, 3, 5, 5]
        check_list_io(i, o, self, siv)

        # Case6:
        i = [-0, -1, -2, -3, -4, -5]
        o = [-0, -1, -2, -3, -4, -5]
        siv = SteppedIntVar(starting=4, step=1)
        check_list_io(i, o, self, siv)

        # Case7: odd list
        siv = SteppedIntVar(starting=9, step=2)
        i = [1, 3, 5, 7, 9, 11]
        o = [1, 3, 5, 7, 9, 11]
        check_list_io(i, o, self, siv)
        i = [2, 4, 6, 8, 10]
        o = [3, 5, 7, 9, 11]
        check_list_io(i, o, self, siv)
        i = [1, 2, 3, 4, 5]
        o = [1, 3, 3, 5, 5]
        check_list_io(i, o, self, siv)

        # Case8:
        i = [-0, -1.1, -2.1, -3.9, -4.1, -5]
        o = [-0, -1  , -2  , -4  , -4  , -5]
        siv = SteppedIntVar(starting=4, step=1)
        check_list_io(i, o, self, siv)


if __name__ == '__main__':
    unittest.main()
