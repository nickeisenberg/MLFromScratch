class Foo:
    stop = 0
    num_if_calls = 0 
    def bar(self):
        """
        calling bar will tigger the if statement twice.
        recursion 0: foo.bar, stop = 1, first bar is called inside the loop
        recursion 1: stop = 2, first bar is called inside the loop
        recursion 2: stop = 3, if statement fails. return to recursion 1. 
        recursion 1: second bar is called. stop = 4. if statement fails.
                     return to recursion 1
        recursion 1: third bar is called. stop = 5. if statement fails.
                     return to recursion 0
        recurion 0: second bar is called. stop = 6. if statement fails.
                    return to recursion 0.
        recursion 0: third bar is called/ stop = 7 if statement fails.
        """
        self.stop += 1
        print(self.stop)
        if self.stop < 3:
            self.num_if_calls += 1
            print('first')
            self.bar()
            print('second')
            self.bar()
            print('third')
            self.bar()
            return 'a'
        else:
            print('if condition not met')
        return self.num_if_calls

foo = Foo()
foo.bar()
