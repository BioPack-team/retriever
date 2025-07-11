from retriever.utils.general import Singleton


def test_singleton() -> None:
    """Test that singleton behavior is correct."""

    class TestClass(metaclass=Singleton):
        some_data: str = ":3"

    a = TestClass()
    b = TestClass()

    assert a is b
