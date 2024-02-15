from types import MethodType
from typing import TypeVar, Iterable
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from tqdm import tqdm as tqdm_std
from tqdm.notebook import tqdm as tqdm_notebook

T = TypeVar('T')

class ProgressManager(AbstractContextManager, ABC):
    def __enter__(self):
        global progress_manager
        if progress_manager is not default_progress_manager:
            raise 'A progress manager has already been initialized'
        progress_manager = self

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        global progress_manager
        progress_manager = default_progress_manager

        return None

    @abstractmethod
    def iterator(self, iterable: Iterable[T], desc: str = None, total: int = None) -> Iterable[T]:
        raise NotImplementedError()

    def updater(self, total: int = None, desc: str = None):
        raise NotImplementedError()

class NullProgress(ProgressManager):
    def iterator(self, iterable: Iterable[T], desc: str = None) -> Iterable[T]:
        return iterable
    
    def updater(self, total: int = None, desc: str = None):
        class Updater:
            @staticmethod
            def update(count: int):
                pass

            def reset(self, n):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, exc_tb):
                pass
        
        return Updater()

class TqdmProgress(ProgressManager, ABC):
    level = -1

    @abstractmethod
    def tqdm(self, *args, **kwargs) -> tqdm_std:
        raise NotImplementedError()

    def iterator(self, iterable: Iterable[T], desc: str = None, total: int = None) -> Iterable[T]:
        self.level += 1
        for v in self.tqdm(iterable, desc=desc, total=total):
            yield v
        self.level -= 1
    
    def updater(self, total: int = None, desc: str = None):
        self.level += 1

        this = self
        class Updater(AbstractContextManager):
            def __init__(self, t: tqdm_std):
                self.t = t

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, exc_tb):
                self.t.close()
                this.level -= 1

            def reset(self, n):
                self.t.reset(n)

            def update(self, count: int):
                self.t.update(count)

        return Updater(self.tqdm(total=total, desc=desc))

class NotebookProgress(TqdmProgress):
    container = None
    bars = []

    def tqdm(self, *args, **kwargs):
        t = tqdm_notebook(*args, **kwargs, display=False, leave=self.level == 0)

        # We do extra work here instead of just setting leave=False because leave=False
        # is broken in VS Code: https://github.com/microsoft/vscode-jupyter/issues/9397
        from IPython.display import display
        if self.container == None or self.level == 0:
            from ipywidgets import VBox
            class WrapperVBox(VBox):
                # Provide alternate text to use when saving the text/plain version of this widget
                # so that when viewing an already-run notebook, it will show something
                # meaningful instead of "VBox()")
                def __repr__(self):
                    nonlocal kwargs
                    return f'Computing {kwargs["desc"]}...'
            self.container = WrapperVBox()
            display(self.container)

        # And a hack for https://github.com/jupyter-widgets/ipywidgets/issues/2585
        old_close = t.container.close
        this = self
        def new_close(self):
            nonlocal this
            if self in this.bars: this.bars.remove(self)
            old_close()
        t.container.close = MethodType(new_close, t.container)

        if self.level == 0:
            self.bars = []

        self.bars.append(t.container)
        self.container.children = self.bars

        return t

class ConsoleProgress(TqdmProgress):
    def tqdm(self, *args, **kwargs):
        return tqdm_std(*args, **kwargs, leave=self.level == 0)

default_progress_manager = NullProgress()
progress_manager = default_progress_manager

def get_progress_manager():
    return progress_manager
