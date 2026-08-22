import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
model = AutoModel.from_pretrained("microsoft/unixcoder-base")

codes = [
    """def get_user_by_id(user_id, users):
    for user in users:
        if user["id"] == user_id:
            return user
    return None""",

    """def find_user(user_id, users):
    return next(
        (user for user in users if user.get("id") == user_id),
        None
    )"""
]

inputs = tokenizer(
    codes,
    max_limit=500,
    return_tensors="pt",
    padding=True,
)


print(type(inputs))
print(type(inputs[0]))

print(len(inputs))
    