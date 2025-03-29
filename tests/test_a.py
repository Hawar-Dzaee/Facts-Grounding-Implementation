import pytest 
from src.utils import fill_out_prompt






# @pytest.mark.parametrize("prompt_template , expected",
#     [
#         ("Hello {{name}}",{"name":"John"},("Hello John")),
#     ]
# )
# def test_fill_out_prompt(prompt_template,expected):
#     assert fill_out_prompt(prompt_template,**expected) == expected


def test_fill_out_prompt():
    # Test basic single variable replacement
    template = """
    Hello {{planet}} Hello {{name}} Hello {{age}} This sign`` shoudln't be here {this is not a placeholder}
    neither should this {{time}}"""     
    result = fill_out_prompt(template, planet="Earth", name="John", age=30,time="10:00")
    assert result == """Hello Earth Hello John Hello 30 This sign`` shoudln't be here {this is not a placeholder}
    neither should this 10:00""" 





