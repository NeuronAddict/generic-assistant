from chat_entry import History, ChatEntry, Role


def test_serialize():

    history = History("hello")
    history.add(ChatEntry(Role.USER, "hello all"))
    assert history.json() == [
        {'content': 'hello', 'name': None, 'role': 'system'},
        {'content': 'hello all', 'name': None, 'role': 'user'},
    ]

def test_double_deserialize():

    history = History("hello")
    history.add(ChatEntry(Role.USER, "hello all"))
    assert history.json() == [
        {'content': 'hello', 'name': None, 'role': 'system'},
        {'content': 'hello all', 'name': None, 'role': 'user'},
    ]
    assert history.json() == [
        {'content': 'hello', 'name': None, 'role': 'system'},
        {'content': 'hello all', 'name': None, 'role': 'user'},
    ]