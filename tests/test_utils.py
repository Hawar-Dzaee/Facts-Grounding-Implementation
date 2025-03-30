import pytest 
import os
from src.utils import fill_out_prompt,dump_in_jsonl





def test_fill_out_prompt():
    template = """My name is {{name}}, I am {{age}} years old.
    `This is not a placeholder` **neither is this** 
    {"sentence" : "Even though it is in a dictionary it is not a placeholder"}"""

    result = fill_out_prompt(template,name="Hawar",age=30)

    assert result == """My name is Hawar, I am 30 years old.
    `This is not a placeholder` **neither is this** 
    {"sentence" : "Even though it is in a dictionary it is not a placeholder"}"""


def test_dump_in_jsonl():
    data = {"name":"Hawar","age":30}
    dump_in_jsonl(data,"test.jsonl")
    assert os.path.exists("test.jsonl")





